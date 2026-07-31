"""Shared synthesis service used by both the HTTP and Wyoming adapters."""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import threading
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, AsyncIterator

import numpy as np
from sentence_stream import SentenceBoundaryDetector

from .audio import audio_to_pcm16_bytes, iter_byte_chunks
from .config import ServerConfig
from .hf_cache import is_huggingface_offline_mode_enabled
from .model_catalog import resolve_model_id
from .prefetch_manifest import resolve_cached_model_snapshot_path
from .voice_registry import VoiceEntry, VoiceRegistry

LOGGER = logging.getLogger(__name__)
_HARDCODED_MODEL_ALIAS = "omnivoice"
_STREAMING_PCM_CHUNK_BYTES = 8192


@dataclass(frozen=True)
class CachedVoiceClonePrompt:
    """CPU-resident OmniVoice prompt object cached for one validated voice entry."""

    voice_id: str
    prompt: object


@dataclass(frozen=True)
class SynthesisRequest:
    """Normalized synthesis request shared across protocol adapters."""

    text: str
    voice_id: str | None = None
    language: str | None = None
    speed: float | None = None



def move_prompt_value_to_cpu(value: object) -> object:
    """Recursively copy tensor-bearing prompt values onto CPU memory.

    Usage:
        OmniVoice prompt objects may contain GPU tensors even though only one
        request can actively generate at a time. This helper converts cached
        prompt payloads into CPU-resident structures so previously used voices
        can be reused later without continuing to pin VRAM between requests.

    Parameters:
        value: Any prompt payload object, including dataclasses, dictionaries,
            lists, tuples, tensors, or primitive values.

    Returns:
        A deep-copied prompt payload where any tensor leaves have been detached
        and moved to CPU memory. Non-tensor values are preserved.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a runtime dependency in production.
        torch = None

    if torch is not None and torch.is_tensor(value):
        return value.detach().cpu()

    if is_dataclass(value) and not isinstance(value, type):
        return value.__class__(
            **{
                field.name: move_prompt_value_to_cpu(getattr(value, field.name))
                for field in fields(value)
            }
        )

    if isinstance(value, dict):
        return {
            key: move_prompt_value_to_cpu(nested_value)
            for key, nested_value in value.items()
        }

    if isinstance(value, list):
        return [move_prompt_value_to_cpu(item) for item in value]

    if isinstance(value, tuple):
        return tuple(move_prompt_value_to_cpu(item) for item in value)

    return value



def normalize_omnivoice_language(language: str | None) -> str | None:
    """Normalize user-supplied language values into OmniVoice-friendly identifiers.

    Usage:
        The HTTP and Wyoming adapters may receive human-readable names, BCP47
        tags, or special auto-detection markers. This helper keeps that cleanup
        logic in one place before values are passed into OmniVoice generation.

    Parameters:
        language: The raw optional language value supplied by a client or config.

    Returns:
        `None` when OmniVoice should auto-detect the language, otherwise a
        cleaned language code or language name string.
    """
    if language is None:
        return None

    normalized_language = language.strip()
    if not normalized_language:
        return None

    normalized_key = normalized_language.replace("_", "-").lower()
    if normalized_key in {"*", "any", "auto", "automatic", "default", "mul"}:
        return None

    if "-" in normalized_key:
        primary_subtag = normalized_key.split("-", 1)[0]
        if primary_subtag:
            return primary_subtag

    return normalized_language



def normalize_omnivoice_device_map(device: str) -> str:
    """Normalize configured device strings into values accepted by OmniVoice.

    Usage:
        Existing deployments often use the shorthand `cuda`, while OmniVoice
        examples and Hugging Face device-map handling are more predictable when a
        concrete CUDA ordinal such as `cuda:0` is used. This helper preserves
        existing non-CUDA values and upgrades the common shorthand.

    Parameters:
        device: The raw runtime device string from configuration.

    Returns:
        A normalized device-map string suitable for `OmniVoice.from_pretrained`.
    """
    normalized_device = device.strip()
    if normalized_device == "cuda":
        return "cuda:0"
    return normalized_device



def coerce_waveform_array(waveform: Any) -> np.ndarray:
    """Convert an OmniVoice waveform output into a flat float32 NumPy array.

    Usage:
        OmniVoice is documented to return NumPy arrays, but tests and future
        library versions may hand back tensor-like objects. This helper gives the
        service one place to coerce those values into the stable mono array shape
        expected by the HTTP and Wyoming adapters.

    Parameters:
        waveform: The first audio item returned by `OmniVoice.generate(...)`.

    Returns:
        A one-dimensional float32 NumPy waveform ready for PCM/WAV conversion.
    """
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu().numpy()
    return np.asarray(waveform, dtype=np.float32).flatten()



def split_text_for_streaming(text: str, max_length: int = 400) -> list[str]:
    """Split request text into sentence-first synthesis segments with a length cap.

    Usage:
        OmniVoice currently exposes buffered generation rather than a documented
        model-incremental audio streaming API. The wrapper splits text into
        sentence-like segments so each synthesis call stays bounded in memory
        and time. Any single segment that exceeds ``max_length`` characters is
        broken further on word boundaries to prevent runaway resource usage.

    Parameters:
        text: The already-validated request text that should be segmented.
        max_length: Maximum character count for any single segment. Sentence
            boundaries are respected first; only oversized segments are split
            further on spaces.

    Returns:
        A non-empty ordered list of stripped text segments suitable for
        sequential synthesis.
    """
    sentence_detector = SentenceBoundaryDetector()
    raw_segments = [
        segment.strip()
        for segment in sentence_detector.add_chunk(text)
        if segment.strip()
    ]
    trailing_segment = sentence_detector.finish().strip()
    if trailing_segment:
        raw_segments.append(trailing_segment)

    if not raw_segments:
        return [text.strip()]

    capped: list[str] = []
    for segment in raw_segments:
        if len(segment) <= max_length:
            capped.append(segment)
        else:
            capped.extend(_split_segment_on_words(segment, max_length))
    return capped


def _split_segment_on_words(segment: str, max_length: int) -> list[str]:
    """Break an oversized segment into smaller chunks on word boundaries.

    Usage:
        ``split_text_for_streaming`` calls this helper when a single sentence
        exceeds the configured character cap so it can still be synthesized
        in manageable pieces without breaking words mid-character.

    Parameters:
        segment: A stripped text segment that is longer than ``max_length``.
        max_length: The maximum character count for each produced chunk.

    Returns:
        An ordered list of stripped sub-segments, each at most ``max_length``
        characters long.
    """
    words = segment.split()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        word_len = len(word)
        # If a single word exceeds the cap, force-split it
        if word_len > max_length:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            # Hard-split the oversized word
            for i in range(0, word_len, max_length):
                chunks.append(word[i : i + max_length])
            continue

        if current_len + (1 if current else 0) + word_len > max_length:
            chunks.append(" ".join(current))
            current = [word]
            current_len = word_len
        else:
            current.append(word)
            current_len += (1 if len(current) > 1 else 0) + word_len

    if current:
        chunks.append(" ".join(current))
    return chunks


class TtsService:
    """Long-lived model wrapper that serializes GPU inference and voice resolution."""

    def __init__(self, config: ServerConfig, voice_registry: VoiceRegistry) -> None:
        """Initialize the shared synthesis service without loading the model yet.

        Usage:
            Construct the service once during application startup, then call
            `await start()` before serving any traffic.

        Parameters:
            config: Immutable runtime configuration for the server.
            voice_registry: The validated voice registry shared by all protocols.

        Returns:
            None. A new `TtsService` instance is initialized in place.
        """
        self.config = config
        self.voice_registry = voice_registry
        self.model_id = resolve_model_id(_HARDCODED_MODEL_ALIAS)
        self.device_map = normalize_omnivoice_device_map(config.device)
        self._model = None
        self._model_lock = threading.Lock()
        self._prepared_voice_lock = threading.Lock()
        self._prepared_voice_cache: dict[str, CachedVoiceClonePrompt] = {}
        self._last_generated_voice_id: str | None = None

    def close(self) -> None:
        """Release cached prompt state and unused accelerator memory.

        Usage:
            Tests or future shutdown hooks can call this method during teardown
            so cached prompt objects are dropped and any unused accelerator cache
            is released before the process exits.

        Parameters:
            None.

        Returns:
            None. Cached prompt objects are cleared in place and best-effort
            accelerator cleanup is performed.
        """
        with self._prepared_voice_lock:
            self._prepared_voice_cache.clear()

        with self._model_lock:
            self._last_generated_voice_id = None
            self._release_unused_accelerator_memory()

    @property
    def sample_rate(self) -> int:
        """Return the model sample rate after startup has loaded the model.

        Usage:
            Protocol adapters use this property to advertise audio metadata and
            construct WAV headers without duplicating model-specific knowledge.

        Parameters:
            None.

        Returns:
            The integer waveform sample rate reported by the loaded model.

        Raises:
            RuntimeError: If startup has not loaded the model yet.
        """
        return int(self._require_model().sampling_rate)

    async def start(self) -> None:
        """Load the OmniVoice model before either server adapter starts serving.

        Usage:
            Call this exactly once during process startup before either server
            adapter begins accepting requests. Voice-clone prompts are built
            lazily on first use so large voice directories do not eagerly occupy
            CPU or GPU memory at startup, unless a specific voice is configured
            for pre-loading via ``--preload-voice`` / ``FATTERVOICE_PRELOAD_VOICE``.

        Parameters:
            None.

        Returns:
            None. The method completes when the model is ready to serve requests.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_model)

        if self.config.preload_voice:
            voice = self.voice_registry.get(self.config.preload_voice)
            await loop.run_in_executor(
                None, lambda: self._resolve_cached_voice_clone_prompt(voice)
            )
            LOGGER.info(
                "Pre-loaded voice clone prompt for voice %s", self.config.preload_voice
            )

    def validate_request(self, request: SynthesisRequest) -> VoiceEntry:
        """Validate request text and resolve the referenced voice before synthesis.

        Usage:
            Protocol adapters call this method before opening a streaming response
            so client-facing validation errors are raised early and consistently.

        Parameters:
            request: The normalized request that should be validated.

        Returns:
            The resolved `VoiceEntry` that should be used for synthesis.

        Raises:
            ValueError: If the request text is empty after surrounding whitespace is removed.
            VoiceRegistryError: If the requested voice does not exist.
        """
        self._validate_request_text(request.text)
        return self.voice_registry.get(request.voice_id)

    async def synthesize(self, request: SynthesisRequest) -> tuple[np.ndarray, int]:
        """Generate a complete waveform for a single synthesis request.

        Usage:
            Non-streaming HTTP responses and Wyoming non-streaming synthesis use
            this method when the entire waveform is needed before returning a
            response. The text is split into sentence-sized segments internally
            so no single OmniVoice call exceeds the configured segment cap.

        Parameters:
            request: A normalized request describing the text, voice, and runtime
                options for the generation.

        Returns:
            A tuple of `(waveform, sample_rate)` where the waveform is a mono
            float32 NumPy array containing all synthesized segments concatenated.
        """
        voice = self.validate_request(request)
        segments = split_text_for_streaming(request.text, self.config.max_sentence_length)
        loop = asyncio.get_running_loop()
        waveforms: list[np.ndarray] = []
        for i, segment_text in enumerate(segments):
            seg_request = SynthesisRequest(
                text=segment_text,
                voice_id=voice.voice_id,
                language=request.language,
                speed=request.speed,
            )
            waveform, _ = await loop.run_in_executor(
                None, lambda sr=seg_request, v=voice: self._generate_waveform(sr, v)
            )
            waveforms.append(waveform)
        combined = np.concatenate(waveforms) if waveforms else np.array([], dtype=np.float32)
        return combined, self.sample_rate

    def stream_pcm_chunks(self, request: SynthesisRequest) -> AsyncIterator[bytes]:
        """Yield PCM chunks by synthesizing sentence-sized segments sequentially.

        Usage:
            All streaming paths (HTTP and Wyoming) use this method. It splits
            validated text into sentence-like segments, synthesizes them one at
            a time, and emits PCM bytes as each segment completes.

        Parameters:
            request: A normalized request describing the text, voice, and runtime
                options for the generation.

        Returns:
            An async iterator that yields raw 16-bit PCM audio chunks as each
            synthesis segment completes.
        """
        resolved_voice = self.validate_request(request)
        segments = split_text_for_streaming(request.text, self.config.max_sentence_length)

        async def emit_pcm_chunks() -> AsyncIterator[bytes]:
            """Synthesize segments sequentially and yield PCM chunks.

            Usage:
                The outer method performs eager validation and segment planning
                so this nested generator focuses on sequential synthesis and
                byte emission once the response body has started.

            Parameters:
                None. It closes over the validated request state.

            Returns:
                An async iterator that yields fixed-size PCM byte chunks.
            """
            loop = asyncio.get_running_loop()
            for segment_text in segments:
                seg_request = SynthesisRequest(
                    text=segment_text,
                    voice_id=resolved_voice.voice_id,
                    language=request.language,
                    speed=request.speed,
                )
                waveform, _ = await loop.run_in_executor(
                    None,
                    lambda sr=seg_request, v=resolved_voice: self._generate_waveform(sr, v),
                )
                pcm_payload = audio_to_pcm16_bytes(waveform)
                for pcm_chunk in iter_byte_chunks(pcm_payload, _STREAMING_PCM_CHUNK_BYTES):
                    yield pcm_chunk

        return emit_pcm_chunks()

    def stream_low_latency_pcm_chunks(self, request: SynthesisRequest) -> AsyncIterator[bytes]:
        """Alias for ``stream_pcm_chunks`` — all paths now use sentence-based splitting.

        Usage:
            The HTTP adapter still calls this name for explicit ``stream=true``
            requests, but the underlying behavior is identical to the default
            streaming path.

        Parameters:
            request: A normalized request describing the text, voice, and runtime
                options for the generation.

        Returns:
            An async iterator that yields raw 16-bit PCM audio chunks.
        """
        return self.stream_pcm_chunks(request)
        resolved_voice = self.validate_request(request)
        streaming_segments = split_text_for_streaming(request.text)

        async def emit_low_latency_pcm_chunks() -> AsyncIterator[bytes]:
            """Generate sentence-sized waveform segments and yield them as PCM chunks.

            Usage:
                The outer method performs eager validation and segment planning so
                the nested generator can focus on sequential segment synthesis and
                byte emission once the HTTP response body has started.

            Parameters:
                None. It closes over the validated request and resolved voice.

            Returns:
                An async iterator that yields fixed-size PCM byte chunks from each
                synthesized text segment in order.
            """
            loop = asyncio.get_running_loop()
            for segment_text in streaming_segments:
                segment_request = SynthesisRequest(
                    text=segment_text,
                    voice_id=resolved_voice.voice_id,
                    language=request.language,
                    speed=request.speed,
                )
                waveform, sample_rate = await loop.run_in_executor(
                    None,
                    lambda: self._generate_waveform(segment_request, resolved_voice),
                )
                _ = sample_rate
                pcm_payload = audio_to_pcm16_bytes(waveform)
                for pcm_chunk in iter_byte_chunks(pcm_payload, _STREAMING_PCM_CHUNK_BYTES):
                    yield pcm_chunk

        return emit_low_latency_pcm_chunks()

    def _load_model(self) -> None:
        """Load the OmniVoice model using the configured device and dtype.

        Usage:
            Startup calls this method through `start()`. Repeated calls are cheap
            because the method exits immediately once the model is already loaded.

        Parameters:
            None.

        Returns:
            None. The loaded model is stored on the service instance.
        """
        if self._model is not None:
            return

        import torch
        from omnivoice import OmniVoice

        try:
            dtype = getattr(torch, self.config.dtype)
        except AttributeError as exc:
            raise ValueError(
                f"Unsupported torch dtype {self.config.dtype!r}."
            ) from exc

        model_source = resolve_cached_model_snapshot_path(self.model_id, None)
        resolved_model_path = Path(model_source).expanduser()

        if is_huggingface_offline_mode_enabled() and not resolved_model_path.exists():
            build_model_selection_hint = os.environ.get("MODEL_SELECTION_HINT", "").strip()
            if build_model_selection_hint and build_model_selection_hint.lower() != "all":
                try:
                    built_model_id_hint = resolve_model_id(build_model_selection_hint)
                except ValueError:
                    built_model_id_hint = build_model_selection_hint
                if built_model_id_hint != self.model_id:
                    raise FileNotFoundError(
                        "Offline mode is enabled, but this container image was built without the requested OmniVoice model. "
                        f"The image build hint is {build_model_selection_hint!r}, while runtime requested {_HARDCODED_MODEL_ALIAS!r} "
                        f"({self.model_id}). Rebuild the image with --build-arg MODEL_SELECTION=all or --build-arg MODEL_SELECTION={_HARDCODED_MODEL_ALIAS}, "
                        "or run an image that already contains the requested snapshot."
                    )
            raise FileNotFoundError(
                "Offline mode is enabled, but no local snapshot path could be resolved for "
                f"{self.model_id!r}. Check the Hugging Face hub cache contents."
            )

        LOGGER.info(
            "Loading OmniVoice model %s from %s on device %s with dtype %s",
            _HARDCODED_MODEL_ALIAS,
            model_source,
            self.device_map,
            self.config.dtype,
        )
        self._model = OmniVoice.from_pretrained(
            model_source,
            device_map=self.device_map,
            dtype=dtype,
            load_asr=False,
        )
        LOGGER.info("Model ready with sample rate %s Hz", self._model.sampling_rate)

    def _require_model(self):
        """Return the loaded OmniVoice model instance or fail with a clear error.

        Usage:
            Internal helper methods call this before attempting prompt creation or
            generation so the error surface stays consistent if startup ordering is
            incorrect.

        Parameters:
            None.

        Returns:
            The loaded `OmniVoice` model instance.

        Raises:
            RuntimeError: If the model has not been loaded yet.
        """
        if self._model is None:
            raise RuntimeError("The TTS model has not been loaded yet.")
        return self._model

    def _release_unused_accelerator_memory(self) -> None:
        """Release unused accelerator cache after prompt or voice transitions.

        Usage:
            The synthesis service calls this helper when it has already dropped
            Python references to GPU-backed prompt data or when it is switching
            away from a previously generated voice. This keeps large prompt-driven
            VRAM high-water marks from lingering longer than necessary.

        Parameters:
            None.

        Returns:
            None. The helper performs best-effort garbage collection and, when
            available, releases unused accelerator cache blocks back to the
            runtime.
        """
        gc.collect()

        try:
            import torch
        except ImportError:  # pragma: no cover - torch is a runtime dependency in production.
            return

        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
            return

        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache") and torch.mps.is_available():
            torch.mps.empty_cache()

    def _resolve_cached_voice_clone_prompt(
        self,
        voice: VoiceEntry,
    ) -> CachedVoiceClonePrompt:
        """Create or retrieve the cached OmniVoice prompt for one validated voice.

        Usage:
            Buffered and streaming synthesis both call this helper so a voice
            prompt is built lazily on first use and then reused for later
            requests. Cached prompts are stored in CPU memory so previously used
            voices do not keep VRAM pinned between requests, while still avoiding
            repeated reference-audio tokenization when users swap back to an
            earlier voice.

        Parameters:
            voice: The validated voice entry selected for the current request.

        Returns:
            A cached `CachedVoiceClonePrompt` object ready for OmniVoice
            generation calls.
        """
        with self._prepared_voice_lock:
            cached_voice = self._prepared_voice_cache.get(voice.voice_id)
            if cached_voice is not None:
                return cached_voice

            LOGGER.info("Creating lazy OmniVoice prompt cache entry for voice %s", voice.voice_id)
            with self._model_lock:
                prompt = self._require_model().create_voice_clone_prompt(
                    ref_audio=str(voice.audio_path),
                    ref_text=voice.transcript,
                    preprocess_prompt=self.config.preprocess_voice_clone_prompt,
                )

            cpu_prompt = move_prompt_value_to_cpu(prompt)
            del prompt
            self._release_unused_accelerator_memory()
            cached_voice = CachedVoiceClonePrompt(
                voice_id=voice.voice_id,
                prompt=cpu_prompt,
            )
            self._prepared_voice_cache[voice.voice_id] = cached_voice
            return cached_voice

    def _validate_request_text(self, text: str) -> None:
        """Validate request text before it reaches the heavy model inference path.

        Usage:
            Both streaming and non-streaming synthesis call this helper to reject
            empty requests consistently across protocol adapters before any model
            work or voice resolution begins.

        Parameters:
            text: The request text that will be synthesized.

        Returns:
            None. The function succeeds silently when the text is acceptable.

        Raises:
            ValueError: If the text is empty after surrounding whitespace is removed.
        """
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("Synthesis text cannot be empty.")

    def _generate_waveform(
        self,
        request: SynthesisRequest,
        voice: VoiceEntry,
    ) -> tuple[np.ndarray, int]:
        """Generate one full OmniVoice waveform for a validated synthesis request.

        Usage:
            This helper centralizes the per-call interaction with OmniVoice so
            the HTTP adapter and Wyoming adapter both share the same prompt
            caching and generation-parameter logic.

        Parameters:
            request: The normalized request supplying text, language, and speed.
            voice: The validated voice entry selected for the request.

        Returns:
            A tuple of `(waveform, sample_rate)` where the waveform is ready for
            WAV/PCM encoding.
        """
        cached_voice = self._resolve_cached_voice_clone_prompt(voice)
        generation_kwargs = self._build_generation_kwargs(request, voice, cached_voice)
        with self._model_lock:
            if (
                self._last_generated_voice_id is not None
                and self._last_generated_voice_id != voice.voice_id
            ):
                self._release_unused_accelerator_memory()

            audio_list = self._require_model().generate(**generation_kwargs)
            self._last_generated_voice_id = voice.voice_id

        waveform = audio_list[0] if audio_list else np.zeros(0, dtype=np.float32)
        normalized_waveform = coerce_waveform_array(waveform)
        del audio_list
        self._release_unused_accelerator_memory()
        return normalized_waveform, self.sample_rate

    def _build_generation_kwargs(
        self,
        request: SynthesisRequest,
        voice: VoiceEntry,
        cached_voice: CachedVoiceClonePrompt,
    ) -> dict[str, object]:
        """Build the OmniVoice keyword arguments for one synthesis call.

        Usage:
            Internal generation methods call this helper so every request uses
            the same cached voice-clone prompt, the selected voice's optional
            instruct text, and the same server-level OmniVoice tuning defaults
            unless a client overrides request-level speed.

        Parameters:
            request: The normalized request supplying text, language, and speed.
            voice: The validated voice entry selected for this request, including
                any optional instruct text loaded from `<voice>.instruct.txt`.
            cached_voice: The tokenized OmniVoice voice-clone prompt selected for
                this request.

        Returns:
            A dictionary of keyword arguments accepted by `OmniVoice.generate`.
        """
        generation_language = normalize_omnivoice_language(
            request.language or self.config.default_language
        )
        generation_kwargs: dict[str, object] = {
            "text": request.text.strip(),
            "language": generation_language,
            "voice_clone_prompt": cached_voice.prompt,
            "num_step": self.config.num_step,
            "guidance_scale": self.config.guidance_scale,
            "denoise": self.config.denoise,
            "t_shift": self.config.t_shift,
            "position_temperature": self.config.position_temperature,
            "class_temperature": self.config.class_temperature,
            "layer_penalty_factor": self.config.layer_penalty_factor,
            "postprocess_output": self.config.postprocess_output_audio,
        }
        if voice.instruct is not None:
            generation_kwargs["instruct"] = voice.instruct
        if request.speed is not None:
            generation_kwargs["speed"] = request.speed
        return generation_kwargs
