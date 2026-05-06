"""Shared synthesis service used by both the HTTP and Wyoming adapters."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import queue
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Callable

import numpy as np

from .audio import audio_to_pcm16_bytes, iter_byte_chunks
from .config import ServerConfig
from .hf_cache import configure_huggingface_cache, is_huggingface_offline_mode_enabled
from .model_catalog import resolve_model_id
from .prefetch_manifest import resolve_prefetched_model_path
from .quality import (
    PreparedVoiceConditioning,
    apply_output_fade_and_padding,
    clean_generated_audio,
    merge_audio_segments_with_crossfade,
    prepare_voice_conditioning,
    split_audio_tail_for_crossfade,
    split_text_for_longform_synthesis,
)
from .voice_registry import VoiceEntry, VoiceRegistry

LOGGER = logging.getLogger(__name__)
_OUTPUT_FADE_MILLISECONDS = 80
_OUTPUT_PAD_MILLISECONDS = 50
_STREAMING_PCM_CHUNK_BYTES = 8192


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
    voice_id: str | None = None
    language: str | None = None
    chunk_size: int | None = None
    non_streaming_mode: bool | None = None
    append_silence: bool | None = None


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
        self._prepared_voice_lock = threading.Lock()
        self._prepared_voice_cache: dict[str, PreparedVoiceConditioning] = {}
        self._prepared_voice_directory = tempfile.TemporaryDirectory(
            prefix="fatterqwen-voices-"
        )

    def close(self) -> None:
        """Release cached temporary resources owned by the synthesis service.

        Usage:
            Tests or future shutdown hooks can call this method to remove the
            temporary directory that stores prepared prompt audio files.

        Parameters:
            None.

        Returns:
            None. Temporary prompt-cache resources are cleaned up in place.
        """
        self._prepared_voice_directory.cleanup()

    def __del__(self) -> None:
        """Best-effort destructor that cleans up temporary prompt-cache files.

        Usage:
            The runtime currently does not have an explicit shutdown hook for the
            service, so this destructor provides a fallback cleanup path for the
            temporary prepared-voice directory.

        Parameters:
            None.

        Returns:
            None. Cleanup errors are intentionally ignored during finalization.
        """
        try:
            self.close()
        except Exception:
            pass

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
        loop = asyncio.get_running_loop()

        def generate_full_audio() -> tuple[np.ndarray, int]:
            """Run synchronous synthesis for the validated request inside a worker thread.

            Usage:
                The async service method delegates to this nested helper so model
                inference, prompt preparation, and long-form postprocessing do not
                block the event loop.

            Parameters:
                None. It closes over the validated request and voice.

            Returns:
                A tuple of `(waveform, sample_rate)` ready to hand back to the
                caller once synthesis has completed.
            """
            prepared_voice = self._resolve_prepared_voice_conditioning(voice)
            text_chunks = self._plan_request_chunks(request.text)
            if len(text_chunks) == 1:
                waveform, sample_rate = self._generate_model_waveform(
                    text=text_chunks[0],
                    request=request,
                    prepared_voice=prepared_voice,
                )
                return self._apply_final_output_finishing(waveform, sample_rate), sample_rate

            LOGGER.info(
                "Chunking synthesis request for voice %s into %d long-form segments.",
                prepared_voice.voice_id,
                len(text_chunks),
            )
            waveform, sample_rate = self._generate_longform_waveform(
                text_chunks=text_chunks,
                request=request,
                prepared_voice=prepared_voice,
            )
            return waveform, sample_rate

        return await loop.run_in_executor(None, generate_full_audio)

    def stream_pcm_chunks(self, request: SynthesisRequest) -> AsyncIterator[bytes]:
        """Yield PCM chunks as the model produces streaming audio for a request.

        Usage:
            The HTTP WAV/PCM endpoint and the Wyoming adapter call this method to
            forward low-latency audio without waiting for the entire utterance to
            finish synthesizing. Request validation runs immediately so protocol
            adapters can reject bad input before opening a stream, while heavier
            voice preparation still starts only when the iterator is consumed.

        Parameters:
            request: A normalized request describing the text, voice, and runtime
                options for the generation.

        Returns:
            An async iterator that yields raw 16-bit PCM audio chunks.
        """
        voice = self.validate_request(request)

        async def consume_streaming_audio() -> AsyncIterator[bytes]:
            """Prepare the voice, run background synthesis, and yield queued PCM.

            Usage:
                The outer method performs eager request validation only, then
                this nested generator handles voice preparation, worker-thread
                orchestration, and queue draining once the caller actually starts
                consuming the stream.

            Parameters:
                None. It closes over the validated request state.

            Returns:
                An async iterator that yields queued PCM chunks until synthesis
                completes or fails.
            """
            prepared_voice = self._resolve_prepared_voice_conditioning(voice)
            text_chunks = self._plan_request_chunks(request.text)
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
                        flag has not been observed yet. This is used for the final
                        completion marker while a consumer is still actively draining
                        the queue.

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
                        if stop_event.is_set():
                            return False

            def produce_streaming_audio() -> None:
                """Run the selected streaming strategy inside a background thread.

                Usage:
                    Short requests keep the vendored low-latency upstream streaming
                    path, while long-form chunked requests use chunk-level generation
                    and boundary smoothing before PCM bytes are queued.

                Parameters:
                    None. It closes over the validated request state.

                Returns:
                    None. PCM chunks and terminal markers are delivered through the queue.
                """
                try:
                    if len(text_chunks) == 1 and not self.config.postprocess_output_audio:
                        for pcm_chunk in self._stream_short_request_pcm_chunks(
                            request=request,
                            prepared_voice=prepared_voice,
                        ):
                            if stop_event.is_set():
                                break
                            if not push_queue_item(pcm_chunk):
                                break
                    else:
                        if len(text_chunks) > 1:
                            LOGGER.info(
                                "Chunking streaming request for voice %s into %d long-form segments.",
                                prepared_voice.voice_id,
                                len(text_chunks),
                            )
                            pcm_chunk_iterator = self._stream_longform_pcm_chunks(
                                text_chunks=text_chunks,
                                request=request,
                                prepared_voice=prepared_voice,
                            )
                        else:
                            pcm_chunk_iterator = self._stream_buffered_request_pcm_chunks(
                                request=request,
                                prepared_voice=prepared_voice,
                            )

                        for pcm_chunk in pcm_chunk_iterator:
                            if stop_event.is_set():
                                break
                            if not push_queue_item(pcm_chunk):
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

        return consume_streaming_audio()

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
            build_model_selection_hint = os.environ.get("MODEL_SELECTION_HINT", "").strip()
            if build_model_selection_hint and build_model_selection_hint.lower() != "all":
                try:
                    built_model_id_hint = resolve_model_id(build_model_selection_hint)
                except ValueError:
                    built_model_id_hint = build_model_selection_hint
                if built_model_id_hint != self.model_id:
                    raise FileNotFoundError(
                        "Offline mode is enabled, but this container image was built without the requested model. "
                        f"The image build hint is {build_model_selection_hint!r}, while runtime requested {self.config.model!r} "
                        f"({self.model_id}). Rebuild the image with --build-arg MODEL_SELECTION=all or --build-arg MODEL_SELECTION={self.config.model}, "
                        "or run an image that already contains the requested snapshot."
                    )
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

    def _resolve_prepared_voice_conditioning(
        self,
        voice: VoiceEntry,
    ) -> PreparedVoiceConditioning:
        """Create or retrieve cached prompt assets for one validated voice.

        Usage:
            Both buffered and streaming synthesis call this helper so every
            request benefits from the same reference-audio cleanup and transcript
            normalization without repeatedly reprocessing the original files.

        Parameters:
            voice: The validated voice entry selected for the current request.

        Returns:
            A cached `PreparedVoiceConditioning` object ready to pass into model
            generation calls.
        """
        with self._prepared_voice_lock:
            cached_voice = self._prepared_voice_cache.get(voice.voice_id)
            if cached_voice is not None:
                return cached_voice

            prepared_audio_path = (
                Path(self._prepared_voice_directory.name) / f"{voice.voice_id}.wav"
            )
            prepared_voice = prepare_voice_conditioning(
                voice_id=voice.voice_id,
                source_audio_path=voice.audio_path,
                transcript=voice.transcript,
                output_audio_path=prepared_audio_path,
                target_sample_rate=self.sample_rate,
                preprocess_reference_audio=self.config.preprocess_reference_audio,
                normalize_reference_transcript=self.config.normalize_reference_transcript,
                reference_prompt_target_rms=self.config.reference_prompt_target_rms,
            )
            self._prepared_voice_cache[voice.voice_id] = prepared_voice
            return prepared_voice

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

    def _plan_request_chunks(self, text: str) -> list[str]:
        """Plan one or more synthesis chunks for a request using weighted text size.

        Usage:
            Long requests are more stable when broken into smaller model calls.
            This helper applies the configured long-form heuristic and returns a
            list containing either the original text or multiple chunk strings.

        Parameters:
            text: The user-provided synthesis text.

        Returns:
            A non-empty ordered list of text chunks ready for generation.
        """
        normalized_text = text.strip()
        if not self.config.longform_chunking_enabled:
            return [normalized_text]
        planned_chunks = split_text_for_longform_synthesis(
            normalized_text,
            threshold_units=self.config.longform_chunk_threshold_units,
            target_units=self.config.longform_target_units,
            min_units=self.config.longform_min_units,
        )
        return planned_chunks or [normalized_text]

    def _generate_model_waveform(
        self,
        *,
        text: str,
        request: SynthesisRequest,
        prepared_voice: PreparedVoiceConditioning,
    ) -> tuple[np.ndarray, int]:
        """Generate and clean one waveform chunk using the cached prompt assets.

        Usage:
            This helper centralizes the per-call interaction with the vendored
            Qwen voice-clone model so both single-shot and long-form orchestration
            paths share the same prompt and cleanup logic.

        Parameters:
            text: The specific text chunk to synthesize in this model call.
            request: The normalized request supplying runtime options.
            prepared_voice: The cached prompt assets for the selected voice.

        Returns:
            A tuple of `(cleaned_waveform, sample_rate)` where the waveform is
            ready for later final finishing or long-form merging.
        """
        generation_kwargs = self._build_common_generation_kwargs(
            text=text,
            request=request,
            prepared_voice=prepared_voice,
        )
        with self._model_lock:
            audio_list, sample_rate = self._require_model().generate_voice_clone(
                **generation_kwargs
            )
        waveform = audio_list[0] if audio_list else np.zeros(0, dtype=np.float32)
        cleaned_waveform = clean_generated_audio(
            waveform,
            int(sample_rate),
            postprocess_output_audio=self.config.postprocess_output_audio,
            reference_rms=prepared_voice.reference_rms,
            prompt_rms=prepared_voice.prompt_rms,
        )
        return cleaned_waveform, int(sample_rate)

    def _generate_longform_waveform(
        self,
        *,
        text_chunks: list[str],
        request: SynthesisRequest,
        prepared_voice: PreparedVoiceConditioning,
    ) -> tuple[np.ndarray, int]:
        """Synthesize multiple text chunks and merge them into one final waveform.

        Usage:
            Requests that exceed the configured long-form threshold use this
            helper to preserve better pacing and consistency by running several
            smaller model calls and smoothing their boundaries.

        Parameters:
            text_chunks: The ordered long-form text chunks that should be spoken.
            request: The normalized request supplying runtime options.
            prepared_voice: The cached prompt assets for the selected voice.

        Returns:
            A tuple of `(merged_waveform, sample_rate)` representing the fully
            merged and final-finished long-form output.
        """
        generated_segments: list[np.ndarray] = []
        sample_rate: int | None = None

        for text_chunk in text_chunks:
            waveform, sample_rate = self._generate_model_waveform(
                text=text_chunk,
                request=request,
                prepared_voice=prepared_voice,
            )
            generated_segments.append(waveform)

        merged_waveform = merge_audio_segments_with_crossfade(
            generated_segments,
            int(sample_rate or self.sample_rate),
            crossfade_milliseconds=self.config.longform_crossfade_milliseconds,
            gap_milliseconds=self.config.longform_gap_milliseconds,
        )
        final_waveform = self._apply_final_output_finishing(
            merged_waveform,
            int(sample_rate or self.sample_rate),
        )
        return final_waveform, int(sample_rate or self.sample_rate)

    def _apply_final_output_finishing(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Apply final edge fades and padding to a completed output waveform.

        Usage:
            The service calls this after one full waveform or a merged long-form
            waveform is otherwise complete so the client hears cleaner starts and
            stops without abrupt clicks.

        Parameters:
            waveform: The finished mono waveform that should receive boundary
                treatment.
            sample_rate: The waveform sample rate in Hertz.

        Returns:
            A new mono float32 waveform with final edge finishing applied.
        """
        return apply_output_fade_and_padding(
            waveform,
            sample_rate,
            fade_milliseconds=_OUTPUT_FADE_MILLISECONDS,
            pad_milliseconds=_OUTPUT_PAD_MILLISECONDS,
        )

    def _stream_short_request_pcm_chunks(
        self,
        *,
        request: SynthesisRequest,
        prepared_voice: PreparedVoiceConditioning,
    ):
        """Yield PCM bytes from the vendored low-latency streaming path for short text.

        Usage:
            Short requests can still use upstream codec-level streaming when the
            operator has disabled output postprocessing and wants the lowest
            possible latency for simple requests.

        Parameters:
            request: The normalized request supplying runtime options.
            prepared_voice: The cached prompt assets for the selected voice.

        Returns:
            A synchronous iterator of PCM byte chunks suitable for queueing into
            the async streaming response.
        """
        generation_kwargs = self._build_streaming_generation_kwargs(
            text=request.text.strip(),
            request=request,
            prepared_voice=prepared_voice,
        )
        with self._model_lock:
            for audio_chunk, _sample_rate, _timing in self._require_model().generate_voice_clone_streaming(
                **generation_kwargs
            ):
                yield audio_to_pcm16_bytes(audio_chunk)

    def _stream_buffered_request_pcm_chunks(
        self,
        *,
        request: SynthesisRequest,
        prepared_voice: PreparedVoiceConditioning,
    ):
        """Yield PCM bytes for one request after full waveform cleanup and finishing.

        Usage:
            When output postprocessing is enabled, the service buffers a complete
            short-request waveform so silence compaction and final edge finishing
            can be applied before any bytes are emitted to the client.

        Parameters:
            request: The normalized request supplying runtime options.
            prepared_voice: The cached prompt assets for the selected voice.

        Returns:
            A synchronous iterator of PCM byte chunks derived from the final
            cleaned waveform for the request.
        """
        waveform, sample_rate = self._generate_model_waveform(
            text=request.text.strip(),
            request=request,
            prepared_voice=prepared_voice,
        )
        finalized_waveform = self._apply_final_output_finishing(
            waveform,
            sample_rate,
        )
        for pcm_chunk in iter_byte_chunks(
            audio_to_pcm16_bytes(finalized_waveform),
            _STREAMING_PCM_CHUNK_BYTES,
        ):
            yield pcm_chunk

    def _stream_longform_pcm_chunks(
        self,
        *,
        text_chunks: list[str],
        request: SynthesisRequest,
        prepared_voice: PreparedVoiceConditioning,
    ):
        """Yield PCM bytes for a long-form request using chunk-level smoothing.

        Usage:
            Long requests trade the vendored token-level streaming path for
            chunk-level synthesis so each chunk can be cleaned, merged, and
            smoothed before audio is emitted to the client.

        Parameters:
            text_chunks: The ordered long-form text chunks that should be spoken.
            request: The normalized request supplying runtime options.
            prepared_voice: The cached prompt assets for the selected voice.

        Returns:
            A synchronous iterator of PCM byte chunks that can be streamed to the
            client as each long-form chunk becomes available.
        """
        pending_tail = np.zeros(0, dtype=np.float32)
        sample_rate: int | None = None
        crossfade_samples = max(
            0,
            int(self.sample_rate * self.config.longform_crossfade_milliseconds / 1000),
        )

        for chunk_index, text_chunk in enumerate(text_chunks):
            waveform, sample_rate = self._generate_model_waveform(
                text=text_chunk,
                request=request,
                prepared_voice=prepared_voice,
            )
            if pending_tail.size > 0:
                waveform = merge_audio_segments_with_crossfade(
                    [pending_tail, waveform],
                    sample_rate,
                    crossfade_milliseconds=self.config.longform_crossfade_milliseconds,
                    gap_milliseconds=self.config.longform_gap_milliseconds,
                )

            is_final_chunk = chunk_index == len(text_chunks) - 1
            if is_final_chunk:
                finalized_waveform = self._apply_final_output_finishing(
                    waveform,
                    sample_rate,
                )
                for pcm_chunk in iter_byte_chunks(
                    audio_to_pcm16_bytes(finalized_waveform),
                    _STREAMING_PCM_CHUNK_BYTES,
                ):
                    yield pcm_chunk
                continue

            emit_now_audio, pending_tail = split_audio_tail_for_crossfade(
                waveform,
                crossfade_samples,
            )
            if chunk_index == 0:
                emit_now_audio = self._apply_leading_stream_fade(
                    emit_now_audio,
                    sample_rate,
                )
            if emit_now_audio.size == 0:
                continue
            for pcm_chunk in iter_byte_chunks(
                audio_to_pcm16_bytes(emit_now_audio),
                _STREAMING_PCM_CHUNK_BYTES,
            ):
                yield pcm_chunk

    def _apply_leading_stream_fade(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Apply a small fade-in to the first emitted streaming audio segment.

        Usage:
            Long-form streaming may emit audio before the final chunk has arrived,
            so only the leading edge can be safely finished up front. This helper
            softens that first emitted boundary without touching the held-back tail.

        Parameters:
            waveform: The first emitted waveform fragment.
            sample_rate: The waveform sample rate in Hertz.

        Returns:
            A copy of the waveform fragment with a short fade-in applied.
        """
        faded_waveform = np.asarray(waveform, dtype=np.float32).flatten().copy()
        if faded_waveform.size == 0:
            return faded_waveform

        fade_samples = max(0, int(sample_rate * _OUTPUT_FADE_MILLISECONDS / 1000))
        fade_length = min(fade_samples, faded_waveform.size)
        if fade_length > 0:
            faded_waveform[:fade_length] *= np.linspace(
                0.0,
                1.0,
                fade_length,
                dtype=np.float32,
            )
        return faded_waveform

    def _build_common_generation_kwargs(
        self,
        *,
        text: str,
        request: SynthesisRequest,
        prepared_voice: PreparedVoiceConditioning,
    ) -> dict[str, object]:
        """Build upstream kwargs shared by buffered and chunked generation calls.

        Usage:
            Internal generation methods call this helper so every model request
            uses the same prepared prompt audio, transcript normalization, and
            runtime flag handling.

        Parameters:
            text: The exact text chunk to synthesize in this upstream call.
            request: The normalized request supplying runtime options.
            prepared_voice: The cached prompt assets for the selected voice.

        Returns:
            A dictionary of keyword arguments accepted by upstream synchronous
            voice cloning.
        """
        return {
            "text": text.strip(),
            "language": request.language or self.config.default_language,
            "ref_audio": str(prepared_voice.audio_path),
            "ref_text": prepared_voice.transcript,
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
        *,
        text: str,
        request: SynthesisRequest,
        prepared_voice: PreparedVoiceConditioning,
    ) -> dict[str, object]:
        """Build upstream kwargs for the vendored short-request streaming path.

        Usage:
            The short-request streaming code path extends the common generation
            kwargs with the codec-chunk size accepted by the upstream streaming
            API.

        Parameters:
            text: The exact text to synthesize in the upstream streaming call.
            request: The normalized request supplying runtime options.
            prepared_voice: The cached prompt assets for the selected voice.

        Returns:
            A dictionary of keyword arguments accepted by upstream streaming
            voice cloning.
        """
        generation_kwargs = self._build_common_generation_kwargs(
            text=text,
            request=request,
            prepared_voice=prepared_voice,
        )
        generation_kwargs["chunk_size"] = request.chunk_size or self.config.chunk_size
        return generation_kwargs
