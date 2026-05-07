"""Unit tests for the shared OmniVoice-backed synthesis service."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from fattervoice.config import ServerConfig
from fattervoice.service import SynthesisRequest, TtsService
from fattervoice.voice_registry import VoiceEntry


class FakeVoiceRegistry:
    """Minimal validated voice registry double used by service-layer unit tests."""

    def __init__(self, entries: list[VoiceEntry]) -> None:
        """Store deterministic voice entries for later service lookups.

        Usage:
            Service tests build this registry instead of scanning a real voices
            directory so they can control the exact voice returned for each case.

        Parameters:
            entries: The voice entries that should appear in the registry.

        Returns:
            None. The registry state is stored on the instance.
        """
        self._entries = {entry.voice_id: entry for entry in entries}
        self.default_voice_id = entries[0].voice_id

    def get(self, voice_id: str | None) -> VoiceEntry:
        """Resolve a voice identifier using the same defaulting behavior as production.

        Usage:
            The `TtsService` calls this method during request validation, so the
            fake registry mirrors the production API surface closely.

        Parameters:
            voice_id: The requested public voice identifier, or `None` for the default.

        Returns:
            The matching `VoiceEntry` object.
        """
        return self._entries[voice_id or self.default_voice_id]

    def list_voice_ids(self) -> list[str]:
        """Return the stable sorted list of available voice IDs.

        Usage:
            The service and adapter layers call this during metadata and error
            construction, so the fake registry exposes the same shape.

        Parameters:
            None.

        Returns:
            A sorted list of voice identifiers.
        """
        return sorted(self._entries)

    def values(self) -> list[VoiceEntry]:
        """Return every configured voice entry in stable voice-ID order.

        Usage:
            Startup prompt priming iterates over this method in production, so
            tests use it to mirror that behavior.

        Parameters:
            None.

        Returns:
            A list of `VoiceEntry` objects.
        """
        return [self._entries[voice_id] for voice_id in self.list_voice_ids()]


class FakeOmniVoiceModel:
    """Small fake OmniVoice model used to exercise service orchestration paths."""

    sampling_rate = 24000

    def __init__(self) -> None:
        """Initialize counters that capture how the service calls the fake model.

        Usage:
            Each test creates a fresh fake model so prompt-creation and generation
            call histories start from a clean state.

        Parameters:
            None.

        Returns:
            None. Call tracking state is stored on the instance.
        """
        self.created_prompts: list[dict[str, object]] = []
        self.generated_requests: list[dict[str, object]] = []

    def create_voice_clone_prompt(
        self,
        *,
        ref_audio: str,
        ref_text: str,
        preprocess_prompt: bool,
    ) -> object:
        """Record prompt-creation inputs and return a reusable opaque prompt object.

        Usage:
            The service caches the returned object, so tests can later assert that
            repeated syntheses do not rebuild the prompt unnecessarily.

        Parameters:
            ref_audio: The reference audio path forwarded by the service.
            ref_text: The transcript paired with the reference audio.
            preprocess_prompt: Whether OmniVoice prompt preprocessing was enabled.

        Returns:
            A simple dictionary that acts as a stable fake prompt object.
        """
        prompt = {
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "preprocess_prompt": preprocess_prompt,
        }
        self.created_prompts.append(prompt)
        return prompt

    def generate(self, **kwargs) -> list[np.ndarray]:
        """Record generation kwargs and return one deterministic waveform.

        Usage:
            Service tests inspect the captured kwargs to verify language, speed,
            and cached prompt handling without importing the real OmniVoice stack.

        Parameters:
            **kwargs: The OmniVoice generation kwargs forwarded by the service.

        Returns:
            A one-item list containing a mono float32 waveform.
        """
        self.generated_requests.append(kwargs)
        return [np.full(480, 0.25, dtype=np.float32)]



def build_test_config(temp_dir: str, **overrides: object) -> ServerConfig:
    """Build a `ServerConfig` instance tailored for service-layer unit tests.

    Usage:
        Individual tests use this helper to start from one stable default config
        and override only the specific fields relevant to the current assertion.

    Parameters:
        temp_dir: Temporary directory root used for filesystem-backed config paths.
        **overrides: Any `ServerConfig` field overrides needed by the caller.

    Returns:
        A fully populated `ServerConfig` instance.
    """
    config_values: dict[str, object] = {
        "voices_dir": Path(temp_dir) / "voices",
        "host": "0.0.0.0",
        "port": 8000,
        "model": "omnivoice",
        "device": "cuda:0",
        "dtype": "float16",
        "default_language": "auto",
        "max_text_length": 4000,
        "model_cache_dir": None,
        "prefetch_manifest_path": None,
        "warmup": False,
        "warmup_text": "Hello from fattervoice.",
        "wyoming_enabled": True,
        "wyoming_uri": "tcp://0.0.0.0:10300",
        "wyoming_audio_chunk_samples": 4096,
        "log_level": "INFO",
        "num_step": 16,
        "guidance_scale": 2.0,
        "denoise": True,
        "t_shift": 0.1,
        "position_temperature": 5.0,
        "class_temperature": 0.0,
        "layer_penalty_factor": 5.0,
        "preprocess_voice_clone_prompt": True,
        "postprocess_output_audio": True,
        "audio_chunk_duration": 15.0,
        "audio_chunk_threshold": 30.0,
    }
    config_values.update(overrides)
    return ServerConfig(**config_values)



def build_test_voice_entry(temp_dir: str) -> VoiceEntry:
    """Create one deterministic `VoiceEntry` for service-layer unit tests.

    Usage:
        Tests that do not need the full production scanner still need a realistic
        voice entry containing public IDs, audio paths, and transcripts.

    Parameters:
        temp_dir: Temporary directory root used to construct stable fake paths.

    Returns:
        A `VoiceEntry` object that points at a synthetic voice pair.
    """
    return VoiceEntry(
        voice_id="demo",
        audio_path=Path(temp_dir) / "voices" / "demo.wav",
        transcript_path=Path(temp_dir) / "voices" / "demo.txt",
        transcript="This is the demo reference transcript.",
    )


class TtsServiceTests(unittest.TestCase):
    """Verify prompt caching, model loading, and request translation in `TtsService`."""

    def test_load_model_uses_omnivoice_loader_and_disables_asr(self) -> None:
        """Ensure `_load_model()` forwards the expected OmniVoice load arguments.

        Usage:
            This test patches lightweight fake `torch` and `omnivoice` modules so
            the service can exercise its real model-loading path without importing
            CUDA or the published OmniVoice package.

        Parameters:
            None.

        Returns:
            None. The test asserts on the captured OmniVoice loader inputs.
        """
        captured_call: dict[str, object] = {}

        class FakeLoadedModel:
            """Minimal loaded-model stand-in used by the model-loading test."""

            sampling_rate = 24000

        class FakeOmniVoice:
            """Stub OmniVoice loader that records the arguments it receives."""

            @classmethod
            def from_pretrained(
                cls,
                model_name: str,
                *,
                device_map: str,
                dtype: object,
                load_asr: bool,
            ) -> FakeLoadedModel:
                """Record OmniVoice loader inputs and return a tiny fake model.

                Usage:
                    The service test calls this through `TtsService._load_model()`
                    to verify the wrapper forwards the expected OmniVoice-specific
                    keyword arguments.

                Parameters:
                    model_name: The resolved local model path or Hugging Face model ID.
                    device_map: The runtime device map selected by the service.
                    dtype: The resolved torch dtype object.
                    load_asr: Whether ASR auto-transcription was requested.

                Returns:
                    A fake loaded model exposing the sample-rate attribute needed
                    by the production service.
                """
                captured_call["model_name"] = model_name
                captured_call["device_map"] = device_map
                captured_call["dtype"] = dtype
                captured_call["load_asr"] = load_asr
                return FakeLoadedModel()

        fake_torch_module = types.SimpleNamespace(float16="fake-float16")
        fake_omnivoice_module = types.SimpleNamespace(OmniVoice=FakeOmniVoice)

        with tempfile.TemporaryDirectory() as temp_dir:
            local_model_path = Path(temp_dir) / "resolved-model"
            local_model_path.mkdir()
            config = build_test_config(temp_dir)
            service = TtsService(
                config=config,
                voice_registry=FakeVoiceRegistry([build_test_voice_entry(temp_dir)]),
            )

            with (
                patch("fattervoice.service.configure_huggingface_cache", return_value=None),
                patch("fattervoice.service.resolve_prefetched_model_path", return_value=str(local_model_path)),
                patch("fattervoice.service.is_huggingface_offline_mode_enabled", return_value=False),
                patch.dict(
                    sys.modules,
                    {
                        "torch": fake_torch_module,
                        "omnivoice": fake_omnivoice_module,
                    },
                ),
            ):
                service._load_model()

        self.assertIsNotNone(service._model)
        self.assertEqual(captured_call["model_name"], str(local_model_path))
        self.assertEqual(captured_call["device_map"], "cuda:0")
        self.assertEqual(captured_call["dtype"], "fake-float16")
        self.assertFalse(captured_call["load_asr"])

    def test_load_model_explains_single_model_offline_image_mismatch(self) -> None:
        """Ensure offline startup explains when runtime requests a non-prefetched model.

        Usage:
            Operators may build a lean image with the built-in OmniVoice snapshot
            and later override `FATTERVOICE_MODEL` at runtime to another Hugging
            Face repo. This test verifies that the resulting offline error clearly
            points to the build/runtime mismatch instead of only reporting a cache miss.

        Parameters:
            None.

        Returns:
            None. The test asserts that the raised error mentions both the build
            hint and the requested runtime model.
        """
        fake_torch_module = types.SimpleNamespace(float16="fake-float16")
        fake_omnivoice_module = types.SimpleNamespace(OmniVoice=object)

        with tempfile.TemporaryDirectory() as temp_dir:
            config = build_test_config(temp_dir, model="acme/custom-omnivoice")
            service = TtsService(
                config=config,
                voice_registry=FakeVoiceRegistry([build_test_voice_entry(temp_dir)]),
            )

            with (
                patch("fattervoice.service.configure_huggingface_cache", return_value=None),
                patch("fattervoice.service.resolve_prefetched_model_path", return_value=service.model_id),
                patch("fattervoice.service.is_huggingface_offline_mode_enabled", return_value=True),
                patch.dict(
                    sys.modules,
                    {
                        "torch": fake_torch_module,
                        "omnivoice": fake_omnivoice_module,
                    },
                ),
                patch.dict(os.environ, {"MODEL_SELECTION_HINT": "omnivoice"}, clear=False),
            ):
                with self.assertRaises(FileNotFoundError) as raised_error:
                    service._load_model()

        self.assertIn("MODEL_SELECTION=all", str(raised_error.exception))
        self.assertIn("omnivoice", str(raised_error.exception))
        self.assertIn("acme/custom-omnivoice", str(raised_error.exception))

    def test_synthesize_reuses_cached_voice_prompt_and_normalizes_language(self) -> None:
        """Ensure repeated synthesis requests reuse one cached OmniVoice prompt.

        Usage:
            Voice cloning performance depends heavily on caching the prompt tokens
            derived from the reference audio. This test verifies that the service
            only prepares the prompt once and forwards normalized language / speed
            values during subsequent generation calls.

        Parameters:
            None.

        Returns:
            None. The test asserts on prompt-cache reuse and generation kwargs.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_model = FakeOmniVoiceModel()
            service = TtsService(
                config=build_test_config(temp_dir),
                voice_registry=FakeVoiceRegistry([build_test_voice_entry(temp_dir)]),
            )
            service._model = fake_model

            async def run_test() -> None:
                await service.synthesize(
                    SynthesisRequest(
                        text="Hello there",
                        voice_id="demo",
                        language="en-US",
                        speed=1.25,
                    )
                )
                await service.synthesize(
                    SynthesisRequest(
                        text="Second line",
                        voice_id="demo",
                        language="en-US",
                        speed=1.0,
                    )
                )

            asyncio.run(run_test())

        self.assertEqual(len(fake_model.created_prompts), 1)
        self.assertEqual(len(fake_model.generated_requests), 2)
        self.assertEqual(fake_model.generated_requests[0]["language"], "en")
        self.assertEqual(fake_model.generated_requests[0]["speed"], 1.25)
        self.assertEqual(
            fake_model.generated_requests[0]["voice_clone_prompt"],
            fake_model.generated_requests[1]["voice_clone_prompt"],
        )

    def test_stream_low_latency_pcm_chunks_synthesizes_sentence_segments_separately(self) -> None:
        """Ensure explicit low-latency streaming reuses cached prompts across segments.

        Usage:
            The OpenAI adapter now uses a sentence-segmented streaming path when
            clients explicitly request `stream=true`. This test verifies that the
            service synthesizes each sentence-like segment separately while still
            reusing the same cached OmniVoice voice-clone prompt object.

        Parameters:
            None.

        Returns:
            None. The test asserts on emitted chunks, per-segment generation, and
            prompt reuse across all generated segments.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_model = FakeOmniVoiceModel()
            service = TtsService(
                config=build_test_config(temp_dir),
                voice_registry=FakeVoiceRegistry([build_test_voice_entry(temp_dir)]),
            )
            service._model = fake_model

            async def collect_chunks() -> list[bytes]:
                return [
                    chunk
                    async for chunk in service.stream_low_latency_pcm_chunks(
                        SynthesisRequest(
                            text="Hello world. Second line without drama!",
                            voice_id="demo",
                        )
                    )
                ]

            streamed_chunks = asyncio.run(collect_chunks())

        self.assertTrue(streamed_chunks)
        self.assertEqual(len(fake_model.created_prompts), 1)
        self.assertEqual(len(fake_model.generated_requests), 2)
        self.assertEqual(
            [generation_request["text"] for generation_request in fake_model.generated_requests],
            ["Hello world.", "Second line without drama!"],
        )
        self.assertEqual(
            fake_model.generated_requests[0]["voice_clone_prompt"],
            fake_model.generated_requests[1]["voice_clone_prompt"],
        )

    def test_stream_pcm_chunks_buffers_full_audio_into_pcm(self) -> None:
        """Ensure the service still exposes buffered chunked PCM streaming semantics.

        Usage:
            OmniVoice currently exposes buffered generation rather than a public
            low-level incremental streaming API. This test verifies that the
            quality-first chunked path still performs one full synthesis request
            before yielding PCM bytes.

        Parameters:
            None.

        Returns:
            None. The test asserts that at least one non-empty PCM chunk is
            emitted and that only one OmniVoice generation call was required.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_model = FakeOmniVoiceModel()
            service = TtsService(
                config=build_test_config(temp_dir),
                voice_registry=FakeVoiceRegistry([build_test_voice_entry(temp_dir)]),
            )
            service._model = fake_model

            async def collect_chunks() -> list[bytes]:
                return [
                    chunk
                    async for chunk in service.stream_pcm_chunks(
                        SynthesisRequest(text="Buffered streaming test", voice_id="demo")
                    )
                ]

            streamed_chunks = asyncio.run(collect_chunks())

        self.assertTrue(streamed_chunks)
        self.assertTrue(all(streamed_chunk for streamed_chunk in streamed_chunks))
        self.assertEqual(len(fake_model.generated_requests), 1)


if __name__ == "__main__":
    unittest.main()
