"""Unit tests for service-layer model loading compatibility helpers."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from fatterqwen.config import ServerConfig
from fatterqwen.quality import PreparedVoiceConditioning
from fatterqwen.service import SynthesisRequest, TtsService, build_supported_model_load_kwargs
from fatterqwen.voice_registry import VoiceEntry


class ModelLoadCompatibilityTests(unittest.TestCase):
    """Verify service-layer compatibility with restored faster-qwen3-tts loaders."""

    def test_build_supported_model_load_kwargs_includes_local_files_only_when_supported(self) -> None:
        """Ensure the helper forwards local-only hints when the loader supports them.

        Usage:
            This test guards compatibility with loader implementations that accept
            Hugging Face-style `local_files_only` keyword arguments.

        Parameters:
            None.

        Returns:
            None. The test asserts that all supported keyword arguments are kept.
        """

        def loader(
            model_name: str,
            *,
            device: str = "cuda",
            dtype: object = None,
            local_files_only: bool = False,
        ) -> object:
            return {
                "model_name": model_name,
                "device": device,
                "dtype": dtype,
                "local_files_only": local_files_only,
            }

        load_kwargs = build_supported_model_load_kwargs(
            loader,
            device="cuda:0",
            dtype="bf16",
            local_files_only=True,
        )

        self.assertEqual(
            load_kwargs,
            {
                "device": "cuda:0",
                "dtype": "bf16",
                "local_files_only": True,
            },
        )

    def test_build_supported_model_load_kwargs_omits_local_files_only_when_unsupported(self) -> None:
        """Ensure the helper omits unsupported kwargs for restored upstream loaders.

        Usage:
            This test protects against the regression where `fatterqwen` passed
            `local_files_only` to a restored `faster-qwen3-tts` loader whose
            `from_pretrained(...)` signature does not accept that keyword.

        Parameters:
            None.

        Returns:
            None. The test asserts that only universally supported kwargs remain.
        """

        def loader(model_name: str, *, device: str = "cuda", dtype: object = None) -> object:
            return {
                "model_name": model_name,
                "device": device,
                "dtype": dtype,
            }

        load_kwargs = build_supported_model_load_kwargs(
            loader,
            device="cuda:0",
            dtype="bf16",
            local_files_only=True,
        )

        self.assertEqual(
            load_kwargs,
            {
                "device": "cuda:0",
                "dtype": "bf16",
            },
        )

    def test_load_model_uses_compatible_kwargs_with_restored_upstream_signature(self) -> None:
        """Ensure `_load_model()` succeeds when upstream lacks local_files_only support.

        Usage:
            This test patches in a lightweight fake `faster_qwen3_tts` module so
            the service can exercise its real model-loading path without importing
            CUDA or the actual upstream package.

        Parameters:
            None.

        Returns:
            None. The test asserts that `_load_model()` forwards only supported
            kwargs and stores the returned model instance.
        """
        captured_call: dict[str, object] = {}

        class FakeLoadedModel:
            """Minimal loaded-model stand-in used by the compatibility test."""

            sample_rate = 24000

        class FakeFasterQwen3TTS:
            """Stub upstream loader whose signature intentionally lacks local_files_only."""

            @classmethod
            def from_pretrained(
                cls,
                model_name: str,
                *,
                device: str = "cuda",
                dtype: object = None,
            ) -> FakeLoadedModel:
                """Record loader inputs and return a tiny fake model instance.

                Usage:
                    The compatibility test calls this through `TtsService._load_model()`
                    to verify that unsupported kwargs are not forwarded.

                Parameters:
                    model_name: The resolved local model path or model ID.
                    device: The configured runtime device string.
                    dtype: The resolved torch dtype object.

                Returns:
                    A fake loaded model exposing only the attributes needed by the test.
                """
                captured_call["model_name"] = model_name
                captured_call["device"] = device
                captured_call["dtype"] = dtype
                return FakeLoadedModel()

        fake_torch_module = types.SimpleNamespace(bfloat16="fake-bfloat16")
        fake_loader_module = types.SimpleNamespace(FasterQwen3TTS=FakeFasterQwen3TTS)

        with tempfile.TemporaryDirectory() as temp_dir:
            local_model_path = Path(temp_dir) / "resolved-model"
            local_model_path.mkdir()
            config = ServerConfig(
                voices_dir=Path(temp_dir) / "voices",
                host="0.0.0.0",
                port=8000,
                model="1.7B",
                device="cuda",
                dtype="bfloat16",
                default_language="Auto",
                chunk_size=8,
                append_silence=True,
                non_streaming_mode=False,
                max_text_length=4000,
                model_cache_dir=None,
                prefetch_manifest_path=None,
                warmup=False,
                warmup_text="Hello from fatterqwen.",
                wyoming_enabled=True,
                wyoming_uri="tcp://0.0.0.0:10300",
                wyoming_audio_chunk_samples=4096,
                log_level="INFO",
            )
            service = TtsService(config, Mock())

            with (
                patch("fatterqwen.service.configure_huggingface_cache", return_value=None),
                patch("fatterqwen.service.resolve_prefetched_model_path", return_value=str(local_model_path)),
                patch("fatterqwen.service.is_huggingface_offline_mode_enabled", return_value=False),
                patch.dict(
                    sys.modules,
                    {
                        "torch": fake_torch_module,
                        "faster_qwen3_tts": fake_loader_module,
                    },
                ),
            ):
                service._load_model()

            self.assertIsNotNone(service._model)
            self.assertEqual(captured_call["model_name"], str(local_model_path))
            self.assertEqual(captured_call["device"], "cuda")
            self.assertEqual(captured_call["dtype"], "fake-bfloat16")
            self.assertNotIn("local_files_only", captured_call)
            service.close()

    def test_load_model_explains_single_model_offline_image_mismatch(self) -> None:
        """Ensure offline startup explains when runtime requests a non-prefetched model.

        Usage:
            Operators may build a lean image with one prefetched model and later
            override `FATTERQWEN_MODEL` at runtime. This test verifies that the
            resulting offline error clearly points to the build/runtime mismatch
            instead of only reporting a generic cache miss.

        Parameters:
            None.

        Returns:
            None. The test asserts that the raised error mentions both the build
            hint and the requested runtime model.
        """
        fake_torch_module = types.SimpleNamespace(bfloat16="fake-bfloat16")
        fake_loader_module = types.SimpleNamespace(FasterQwen3TTS=object)

        with tempfile.TemporaryDirectory() as temp_dir:
            config = ServerConfig(
                voices_dir=Path(temp_dir) / "voices",
                host="0.0.0.0",
                port=8000,
                model="0.6B",
                device="cuda",
                dtype="bfloat16",
                default_language="Auto",
                chunk_size=8,
                append_silence=True,
                non_streaming_mode=False,
                max_text_length=4000,
                model_cache_dir=None,
                prefetch_manifest_path=Path(temp_dir) / "prefetched-models.json",
                warmup=False,
                warmup_text="Hello from fatterqwen.",
                wyoming_enabled=True,
                wyoming_uri="tcp://0.0.0.0:10300",
                wyoming_audio_chunk_samples=4096,
                log_level="INFO",
            )
            service = TtsService(config, Mock())

            with (
                patch("fatterqwen.service.configure_huggingface_cache", return_value=None),
                patch("fatterqwen.service.resolve_prefetched_model_path", return_value=service.model_id),
                patch("fatterqwen.service.is_huggingface_offline_mode_enabled", return_value=True),
                patch.dict(
                    sys.modules,
                    {
                        "torch": fake_torch_module,
                        "faster_qwen3_tts": fake_loader_module,
                    },
                ),
                patch.dict(os.environ, {"MODEL_SELECTION_HINT": "1.7B"}, clear=False),
            ):
                with self.assertRaises(FileNotFoundError) as raised_error:
                    service._load_model()
            service.close()

        self.assertIn("MODEL_SELECTION=all", str(raised_error.exception))
        self.assertIn("1.7B", str(raised_error.exception))
        self.assertIn("0.6B", str(raised_error.exception))


class _FakeCloneModel:
    """Small fake synthesis model used to exercise service orchestration paths.

    Usage:
        The long-form service tests install this fake model directly on the
        service instance so they can verify chunk planning and streaming behavior
        without importing CUDA or the real upstream model.

    Parameters:
        None.

    Returns:
        A lightweight object exposing the subset of model methods used by the
        service tests.
    """

    sample_rate = 24000

    def __init__(self) -> None:
        """Initialize counters that capture how the service calls the fake model.

        Usage:
            Each test creates a fresh fake model so call counts and captured text
            chunks start from a clean state.

        Parameters:
            None.

        Returns:
            None. The fake model tracks synchronous and streaming calls in place.
        """
        self.generated_texts: list[str] = []
        self.streaming_call_count = 0

    def generate_voice_clone(self, **kwargs) -> tuple[list[np.ndarray], int]:
        """Return a deterministic waveform whose value encodes the call index.

        Usage:
            The service's buffered and long-form chunked code paths call this
            method exactly as they would the real upstream model.

        Parameters:
            **kwargs: Model-generation keyword arguments, including the `text`
                chunk selected by the service.

        Returns:
            A tuple of `([waveform], sample_rate)` matching the real upstream API.
        """
        text = str(kwargs["text"])
        self.generated_texts.append(text)
        amplitude = 0.2 + (0.05 * len(self.generated_texts))
        waveform = np.full(600, amplitude, dtype=np.float32)
        return [waveform], self.sample_rate

    def generate_voice_clone_streaming(self, **kwargs):
        """Record that streaming was used and yield one deterministic chunk.

        Usage:
            The short-request streaming test path still uses this method, while
            the long-form chunked streaming test asserts that it is not called.

        Parameters:
            **kwargs: Model-generation keyword arguments forwarded by the service.

        Returns:
            A generator yielding one `(waveform, sample_rate, timing)` tuple.
        """
        self.streaming_call_count += 1
        yield np.full(320, 0.25, dtype=np.float32), self.sample_rate, {"chunk_index": 0}


class LongformServiceTests(unittest.TestCase):
    """Verify the wrapper's long-form quality orchestration around the upstream model."""

    def _build_config(self) -> ServerConfig:
        """Create a test config that forces long-form chunking for short sentences.

        Usage:
            The service tests use an intentionally tiny chunk threshold so they can
            exercise multi-call orchestration with small synthetic request text.

        Parameters:
            None.

        Returns:
            A `ServerConfig` tailored for deterministic long-form service tests.
        """
        return ServerConfig(
            voices_dir=Path("/tmp/voices"),
            host="0.0.0.0",
            port=8000,
            model="1.7B",
            device="cuda",
            dtype="bfloat16",
            default_language="Auto",
            chunk_size=8,
            append_silence=True,
            non_streaming_mode=False,
            max_text_length=4000,
            model_cache_dir=None,
            prefetch_manifest_path=None,
            warmup=False,
            warmup_text="Hello from fatterqwen.",
            wyoming_enabled=True,
            wyoming_uri="tcp://0.0.0.0:10300",
            wyoming_audio_chunk_samples=4096,
            log_level="INFO",
            postprocess_output_audio=False,
            longform_chunking_enabled=True,
            longform_chunk_threshold_units=12,
            longform_target_units=10,
            longform_min_units=4,
            longform_crossfade_milliseconds=20,
            longform_gap_milliseconds=30,
        )

    def _build_short_streaming_config(self) -> ServerConfig:
        """Create a test config that keeps one short request in a single chunk.

        Usage:
            The short-streaming quality test needs postprocessing enabled while
            still avoiding the long-form chunk planner, so this helper returns a
            config with a very large chunk threshold.

        Parameters:
            None.

        Returns:
            A `ServerConfig` suited for single-chunk buffered streaming tests.
        """
        return ServerConfig(
            voices_dir=Path("/tmp/voices"),
            host="0.0.0.0",
            port=8000,
            model="1.7B",
            device="cuda",
            dtype="bfloat16",
            default_language="Auto",
            chunk_size=8,
            append_silence=True,
            non_streaming_mode=False,
            max_text_length=4000,
            model_cache_dir=None,
            prefetch_manifest_path=None,
            warmup=False,
            warmup_text="Hello from fatterqwen.",
            wyoming_enabled=True,
            wyoming_uri="tcp://0.0.0.0:10300",
            wyoming_audio_chunk_samples=4096,
            log_level="INFO",
            postprocess_output_audio=True,
            longform_chunking_enabled=True,
            longform_chunk_threshold_units=1000,
            longform_target_units=400,
            longform_min_units=80,
            longform_crossfade_milliseconds=20,
            longform_gap_milliseconds=30,
        )

    def _build_voice_entry(self, temp_dir: str) -> VoiceEntry:
        """Create a minimal validated-style voice entry for service orchestration tests.

        Usage:
            The tests patch prompt preparation, so the voice entry only needs to
            provide stable metadata for request validation.

        Parameters:
            temp_dir: Temporary directory used to hold synthetic file paths.

        Returns:
            A `VoiceEntry` with predictable test paths and transcript text.
        """
        return VoiceEntry(
            voice_id="demo",
            audio_path=Path(temp_dir) / "demo.wav",
            transcript_path=Path(temp_dir) / "demo.txt",
            transcript="Reference transcript.",
        )

    def test_synthesize_chunks_large_requests_into_multiple_model_calls(self) -> None:
        """Ensure buffered synthesis uses multiple upstream calls for long text.

        Usage:
            This test verifies the highest-impact quality change: large requests
            are split into several smaller model calls and then merged instead of
            being forced through one monolithic generation pass.

        Parameters:
            None.

        Returns:
            None. The test asserts that multiple model calls occur and a waveform
            is returned successfully.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            voice_entry = self._build_voice_entry(temp_dir)
            voice_registry = Mock()
            voice_registry.get.return_value = voice_entry
            service = TtsService(self._build_config(), voice_registry)
            service._model = _FakeCloneModel()
            prepared_voice = PreparedVoiceConditioning(
                voice_id="demo",
                audio_path=Path(temp_dir) / "prepared.wav",
                transcript="Reference transcript.",
                reference_rms=0.2,
                prompt_rms=0.2,
            )

            with patch.object(
                service,
                "_resolve_prepared_voice_conditioning",
                return_value=prepared_voice,
            ):
                waveform, sample_rate = asyncio.run(
                    service.synthesize(
                        SynthesisRequest(
                            text="Alpha sentence. Beta sentence. Gamma sentence.",
                            voice_id="demo",
                        )
                    )
                )
            service.close()

        self.assertEqual(sample_rate, 24000)
        self.assertGreater(waveform.shape[0], 0)
        self.assertGreater(len(service._model.generated_texts), 1)

    def test_stream_pcm_chunks_uses_chunked_longform_path_without_upstream_streaming(self) -> None:
        """Ensure long-form streaming uses chunk-level synthesis rather than token streaming.

        Usage:
            The quality-focused streaming path must bypass the upstream token-level
            streamer for chunked long-form requests so cleaned and smoothed audio
            can be emitted instead.

        Parameters:
            None.

        Returns:
            None. The test asserts that PCM bytes are produced, several buffered
            model calls are made, and the upstream streaming generator is skipped.
        """
        async def collect_pcm_chunks(service: TtsService) -> bytes:
            """Collect every emitted PCM chunk into one payload for assertions.

            Usage:
                The test uses this helper to consume the async service iterator in
                a compact, deterministic way.

            Parameters:
                service: The `TtsService` instance under test.

            Returns:
                One concatenated byte payload containing every emitted PCM chunk.
            """
            payload_parts: list[bytes] = []
            async for pcm_chunk in service.stream_pcm_chunks(
                SynthesisRequest(
                    text="Alpha sentence. Beta sentence. Gamma sentence.",
                    voice_id="demo",
                )
            ):
                payload_parts.append(pcm_chunk)
            return b"".join(payload_parts)

        with tempfile.TemporaryDirectory() as temp_dir:
            voice_entry = self._build_voice_entry(temp_dir)
            voice_registry = Mock()
            voice_registry.get.return_value = voice_entry
            service = TtsService(self._build_config(), voice_registry)
            service._model = _FakeCloneModel()
            prepared_voice = PreparedVoiceConditioning(
                voice_id="demo",
                audio_path=Path(temp_dir) / "prepared.wav",
                transcript="Reference transcript.",
                reference_rms=0.2,
                prompt_rms=0.2,
            )

            with patch.object(
                service,
                "_resolve_prepared_voice_conditioning",
                return_value=prepared_voice,
            ):
                pcm_payload = asyncio.run(collect_pcm_chunks(service))
            service.close()

        self.assertGreater(len(pcm_payload), 0)
        self.assertGreater(len(service._model.generated_texts), 1)
        self.assertEqual(service._model.streaming_call_count, 0)

    def test_stream_pcm_chunks_buffers_short_requests_when_output_postprocessing_is_enabled(self) -> None:
        """Ensure short streamed requests use buffered cleanup instead of raw upstream PCM.

        Usage:
            The quality refactor promises that generated audio can be cleaned and
            edge-finished before being returned. This test verifies that enabling
            postprocessing keeps short streamed requests off the raw upstream
            streaming path so that cleanup can be applied first.

        Parameters:
            None.

        Returns:
            None. The test asserts that a short streamed request emits PCM bytes
            without invoking the upstream streaming generator.
        """
        async def collect_pcm_chunks(service: TtsService) -> bytes:
            """Collect the service's async PCM iterator into one payload.

            Usage:
                The test uses this helper to consume the streaming API in one
                place while keeping the assertions focused on orchestration.

            Parameters:
                service: The `TtsService` instance under test.

            Returns:
                The concatenated PCM payload emitted by the service.
            """
            payload_parts: list[bytes] = []
            async for pcm_chunk in service.stream_pcm_chunks(
                SynthesisRequest(
                    text="A short sentence.",
                    voice_id="demo",
                )
            ):
                payload_parts.append(pcm_chunk)
            return b"".join(payload_parts)

        with tempfile.TemporaryDirectory() as temp_dir:
            voice_entry = self._build_voice_entry(temp_dir)
            voice_registry = Mock()
            voice_registry.get.return_value = voice_entry
            service = TtsService(self._build_short_streaming_config(), voice_registry)
            service._model = _FakeCloneModel()
            prepared_voice = PreparedVoiceConditioning(
                voice_id="demo",
                audio_path=Path(temp_dir) / "prepared.wav",
                transcript="Reference transcript.",
                reference_rms=0.2,
                prompt_rms=0.2,
            )

            with patch.object(
                service,
                "_resolve_prepared_voice_conditioning",
                return_value=prepared_voice,
            ):
                pcm_payload = asyncio.run(collect_pcm_chunks(service))
            service.close()

        self.assertGreater(len(pcm_payload), 0)
        self.assertEqual(len(service._model.generated_texts), 1)
        self.assertEqual(service._model.streaming_call_count, 0)


if __name__ == "__main__":
    unittest.main()
