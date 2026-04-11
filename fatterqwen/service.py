"""Shared synthesis service used by both the HTTP and Wyoming adapters."""

from __future__ import annotations

import asyncio
import inspect
import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Callable, Optional

import numpy as np

from .audio import audio_to_pcm16_bytes
from .config import ServerConfig
from .hf_cache import configure_huggingface_cache, is_huggingface_offline_mode_enabled
from .model_catalog import resolve_model_id
from .prefetch_manifest import resolve_prefetched_model_path
from .voice_registry import VoiceEntry, VoiceRegistry

LOGGER = logging.getLogger(__name__)


def build_supported_model_load_kwargs(
    loader: Callable[..., object],
    *,
    device: str,
    dtype: object,
    local_files_only: bool,
) -> dict[str, object]:
    """Build a `from_pretrained` kwargs dictionary compatible with the current loader API.

    Usage:
        Runtime model loading calls this helper before invoking the vendored
        `faster-qwen3-tts` loader so the wrapper stays compatible with restored or
        upgraded copies whose `from_pretrained(...)` signature may or may not
        accept optional Hugging Face-style keyword arguments such as
        `local_files_only`.

    Parameters:
        loader: The callable whose signature should be inspected, usually
            `FasterQwen3TTS.from_pretrained`.
        device: The runtime device string that should be forwarded.
        dtype: The resolved torch dtype object that should be forwarded.
        local_files_only: Whether runtime would prefer stricter local-only model
            resolution when the loader supports that hint.

    Returns:
        A dictionary containing only keyword arguments that the inspected loader
        explicitly supports.
    """
    supported_parameters = inspect.signature(loader).parameters
    load_kwargs: dict[str, object] = {
        "device": device,
        "dtype": dtype,
    }
    if "local_files_only" in supported_parameters:
        load_kwargs["local_files_only"] = local_files_only
    return load_kwargs


@dataclass(frozen=True)
class SynthesisRequest:
    """Normalized synthesis request shared across protocol adapters."""

    text: str
    voice_id: Optional[str] = None
    language: Optional[str] = None
    chunk_size: Optional[int] = None
    non_streaming_mode: Optional[bool] = None
    append_silence: Optional[bool] = None


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
        self.model_id = resolve_model_id(config.model)
        self._model = None
        self._model_lock = threading.Lock()

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
        return int(self._require_model().sample_rate)

    async def start(self) -> None:
        """Load the underlying faster-qwen3-tts model and optionally warm it up.

        Usage:
            Call this exactly once during process startup before either server
            adapter begins accepting requests.

        Parameters:
            None.

        Returns:
            None. The method completes when the model is ready to serve requests.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_model)
        if self.config.warmup:
            await self.warmup()

    async def warmup(self) -> None:
        """Run a short synthesis request so CUDA graph capture happens before traffic.

        Usage:
            This method is optional and is only invoked when startup warmup is
            enabled. It uses the default voice from the validated voice registry.

        Parameters:
            None.

        Returns:
            None. The method completes when the warmup request finishes.
        """
        default_voice = self.voice_registry.get(None)
        LOGGER.info("Running warmup request using voice %s", default_voice.voice_id)
        await self.synthesize(
            SynthesisRequest(
                text=self.config.warmup_text,
                voice_id=default_voice.voice_id,
            )
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
            ValueError: If the request text is empty or exceeds the configured limit.
            VoiceRegistryError: If the requested voice does not exist.
        """
        self._validate_request_text(request.text)
        return self.voice_registry.get(request.voice_id)

    async def synthesize(self, request: SynthesisRequest) -> tuple[np.ndarray, int]:
        """Generate a complete waveform for a single synthesis request.

        Usage:
            Non-streaming HTTP responses and warmup logic use this method when the
            entire waveform is needed before returning a response.

        Parameters:
            request: A normalized request describing the text, voice, and runtime
                options for the generation.

        Returns:
            A tuple of `(waveform, sample_rate)` where the waveform is a mono
            float32 NumPy array.
        """
        voice = self.validate_request(request)
        generation_kwargs = self._build_common_generation_kwargs(request, voice)
        loop = asyncio.get_running_loop()

        def generate_full_audio() -> tuple[np.ndarray, int]:
            """Run a blocking full-audio generation call inside a worker thread.

            Usage:
                This nested helper is executed via the event loop's executor so the
                async protocol adapters do not block while the model runs.

            Parameters:
                None. It closes over the already validated generation inputs.

            Returns:
                A tuple of `(waveform, sample_rate)` produced by the underlying model.
            """
            with self._model_lock:
                audio_list, sample_rate = self._require_model().generate_voice_clone(
                    **generation_kwargs
                )
            waveform = audio_list[0] if audio_list else np.zeros(0, dtype=np.float32)
            return waveform, int(sample_rate)

        return await loop.run_in_executor(None, generate_full_audio)

    async def stream_pcm_chunks(self, request: SynthesisRequest) -> AsyncIterator[bytes]:
        """Yield PCM chunks as the model produces streaming audio for a request.

        Usage:
            The HTTP WAV/PCM endpoint and the Wyoming adapter call this method to
            forward low-latency audio without waiting for the entire utterance to
            finish synthesizing.

        Parameters:
            request: A normalized request describing the text, voice, and runtime
                options for the generation.

        Returns:
            An async iterator that yields raw 16-bit PCM audio chunks.
        """
        voice = self.validate_request(request)
        generation_kwargs = self._build_streaming_generation_kwargs(request, voice)
        item_queue: "queue.Queue[object]" = queue.Queue(maxsize=8)
        finished_marker = object()
        stop_event = threading.Event()

        def push_queue_item(item: object, force: bool = False) -> bool:
            """Push an item into the streaming queue while honoring cancellation.

            Usage:
                The background producer uses this helper to provide bounded
                backpressure and to stop cleanly when the async consumer has gone
                away or cancelled the response.

            Parameters:
                item: The queue item to push.
                force: When true, keep trying to deliver the item even if the stop
                    flag has been set. This is used for the final completion marker.

            Returns:
                `True` when the item was queued successfully, otherwise `False`
                when cancellation prevented delivery.
            """
            while True:
                if stop_event.is_set() and not force:
                    return False
                try:
                    item_queue.put(item, timeout=0.25)
                    return True
                except queue.Full:
                    if stop_event.is_set() and not force:
                        return False

        def produce_streaming_audio() -> None:
            """Run the blocking upstream streaming generator inside a background thread.

            Usage:
                This helper acquires the model lock, iterates the upstream
                generator, converts each waveform chunk to PCM bytes, and pushes
                them into a bounded thread-safe queue consumed by the async caller.

            Parameters:
                None. It closes over the already validated generation inputs.

            Returns:
                None. Audio chunks and terminal markers are delivered through the queue.
            """
            try:
                with self._model_lock:
                    for audio_chunk, _sample_rate, _timing in self._require_model().generate_voice_clone_streaming(
                        **generation_kwargs
                    ):
                        if stop_event.is_set():
                            break
                        if not push_queue_item(audio_to_pcm16_bytes(audio_chunk)):
                            break
            except Exception as exc:  # pragma: no cover - exercised during runtime integration.
                push_queue_item(exc)
            finally:
                push_queue_item(finished_marker, force=True)

        threading.Thread(target=produce_streaming_audio, daemon=True).start()
        loop = asyncio.get_running_loop()

        try:
            while True:
                next_item = await loop.run_in_executor(None, item_queue.get)
                if next_item is finished_marker:
                    break
                if isinstance(next_item, Exception):
                    raise next_item
                yield next_item
        finally:
            stop_event.set()

    def _load_model(self) -> None:
        """Load the upstream CUDA-graph model using the configured device and dtype.

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

        configure_huggingface_cache(self.config.model_cache_dir)

        import torch
        from faster_qwen3_tts import FasterQwen3TTS

        try:
            dtype = getattr(torch, self.config.dtype)
        except AttributeError as exc:
            raise ValueError(
                f"Unsupported torch dtype {self.config.dtype!r}."
            ) from exc

        model_source = resolve_prefetched_model_path(
            self.model_id,
            self.config.prefetch_manifest_path,
            self.config.model_cache_dir,
        )
        resolved_model_path = Path(model_source).expanduser()
        local_files_only = (
            resolved_model_path.exists()
            or is_huggingface_offline_mode_enabled()
        )
        if (
            self.config.prefetch_manifest_path is not None
            and model_source == self.model_id
        ):
            LOGGER.warning(
                "Prefetch manifest %s did not resolve a local snapshot path for %s; falling back to Hugging Face cache inspection / model ID loading.",
                self.config.prefetch_manifest_path,
                self.model_id,
            )
        if is_huggingface_offline_mode_enabled() and not resolved_model_path.exists():
            raise FileNotFoundError(
                "Offline mode is enabled, but no local snapshot path could be resolved for "
                f"{self.model_id!r}. Check FATTERQWEN_PREFETCH_MANIFEST and the Hugging Face hub cache contents."
            )
        load_kwargs = build_supported_model_load_kwargs(
            FasterQwen3TTS.from_pretrained,
            device=self.config.device,
            dtype=dtype,
            local_files_only=local_files_only,
        )
        if local_files_only and "local_files_only" not in load_kwargs:
            LOGGER.info(
                "Current faster-qwen3-tts loader does not accept local_files_only; relying on the resolved local model path and offline environment flags instead."
            )
        LOGGER.info(
            "Loading model %s from %s on device %s with dtype %s",
            self.config.model,
            model_source,
            self.config.device,
            self.config.dtype,
        )
        self._model = FasterQwen3TTS.from_pretrained(model_source, **load_kwargs)
        LOGGER.info("Model ready with sample rate %s Hz", self._model.sample_rate)

    def _require_model(self):
        """Return the loaded upstream model instance or fail with a clear error.

        Usage:
            Internal helper methods call this before attempting generation so the
            error surface stays consistent if startup ordering is incorrect.

        Parameters:
            None.

        Returns:
            The loaded `FasterQwen3TTS` model instance.

        Raises:
            RuntimeError: If the model has not been loaded yet.
        """
        if self._model is None:
            raise RuntimeError("The TTS model has not been loaded yet.")
        return self._model

    def _validate_request_text(self, text: str) -> None:
        """Validate request text before it reaches the heavy model inference path.

        Usage:
            Both streaming and non-streaming synthesis call this helper to reject
            empty or excessively long requests consistently across protocols.

        Parameters:
            text: The request text that will be synthesized.

        Returns:
            None. The function succeeds silently when the text is acceptable.

        Raises:
            ValueError: If the text is empty or longer than the configured limit.
        """
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("Synthesis text cannot be empty.")
        if len(normalized_text) > self.config.max_text_length:
            raise ValueError(
                "Synthesis text exceeds the configured maximum length of "
                f"{self.config.max_text_length} characters."
            )

    def _build_common_generation_kwargs(
        self,
        request: SynthesisRequest,
        voice: VoiceEntry,
    ) -> dict[str, object]:
        """Build the upstream kwargs shared by sync and streaming generation modes.

        Usage:
            Internal generation methods call this helper so both code paths share
            the same text normalization, voice resolution, and runtime option logic.

        Parameters:
            request: The normalized request from the HTTP or Wyoming layer.
            voice: The validated voice entry selected for this request.

        Returns:
            A dictionary of keyword arguments accepted by upstream synchronous voice cloning.
        """
        return {
            "text": request.text.strip(),
            "language": request.language or self.config.default_language,
            "ref_audio": str(voice.audio_path),
            "ref_text": voice.transcript,
            "non_streaming_mode": (
                self.config.non_streaming_mode
                if request.non_streaming_mode is None
                else request.non_streaming_mode
            ),
            "append_silence": (
                self.config.append_silence
                if request.append_silence is None
                else request.append_silence
            ),
        }

    def _build_streaming_generation_kwargs(
        self,
        request: SynthesisRequest,
        voice: VoiceEntry,
    ) -> dict[str, object]:
        """Build the upstream kwargs for streaming generation specifically.

        Usage:
            The streaming code path extends the common generation kwargs with the
            chunk-size knob that is only accepted by the upstream streaming API.

        Parameters:
            request: The normalized request from the HTTP or Wyoming layer.
            voice: The validated voice entry selected for this request.

        Returns:
            A dictionary of keyword arguments accepted by upstream streaming voice cloning.
        """
        generation_kwargs = self._build_common_generation_kwargs(request, voice)
        generation_kwargs["chunk_size"] = request.chunk_size or self.config.chunk_size
        return generation_kwargs
