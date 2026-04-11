"""Unit tests for service-layer model loading compatibility helpers."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fatterqwen.config import ServerConfig
from fatterqwen.service import TtsService, build_supported_model_load_kwargs


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


if __name__ == "__main__":
    unittest.main()
