"""Unit tests for runtime configuration parsing."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fattervoice.config import parse_server_config


class ServerConfigTests(unittest.TestCase):
    """Verify that runtime configuration honors the documented CLI and env contract."""

    def test_parse_server_config_uses_huggingface_cache_environment_fallbacks(self) -> None:
        """Ensure runtime startup can reuse the Docker image's Hugging Face cache env vars.

        Usage:
            The Docker image sets standard Hugging Face cache environment variables
            even when `FATTERVOICE_MODEL_CACHE_DIR` is not explicitly provided.
            This test verifies that runtime config parsing still resolves the same
            cache directory so offline startup can inspect the prefetched cache.

        Parameters:
            None.

        Returns:
            None. The test asserts that the parsed config points at the cache
            directory declared through `HF_HUB_CACHE`.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            voices_dir = Path(temp_dir) / "voices"
            voices_dir.mkdir()
            cache_dir = Path(temp_dir) / "huggingface" / "hub"
            with patch.dict(
                os.environ,
                {
                    "HF_HUB_CACHE": str(cache_dir),
                    "FATTERVOICE_MODEL_CACHE_DIR": "",
                    "HUGGINGFACE_HUB_CACHE": "",
                    "TRANSFORMERS_CACHE": "",
                },
                clear=True,
            ):
                server_config = parse_server_config([
                    "--voices-dir",
                    str(voices_dir),
                ])

            self.assertEqual(server_config.model_cache_dir, cache_dir.resolve())

    def test_parse_server_config_supports_omnivoice_generation_flags(self) -> None:
        """Ensure the OmniVoice runtime knobs can be configured through the CLI.

        Usage:
            The backend migration adds OmniVoice-specific decoding and prompt
            preparation options. This test verifies that representative CLI flags
            land in the parsed server config.

        Parameters:
            None.

        Returns:
            None. The test asserts that the parsed config contains the requested
            OmniVoice settings.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            voices_dir = Path(temp_dir) / "voices"
            voices_dir.mkdir()

            server_config = parse_server_config([
                "--voices-dir",
                str(voices_dir),
                "--num-step",
                "24",
                "--guidance-scale",
                "2.5",
                "--no-denoise",
                "--position-temperature",
                "4.0",
                "--class-temperature",
                "0.2",
                "--layer-penalty-factor",
                "6.0",
                "--no-preprocess-voice-clone-prompt",
                "--no-postprocess-output-audio",
                "--audio-chunk-duration",
                "12.0",
                "--audio-chunk-threshold",
                "24.0",
            ])

        self.assertEqual(server_config.num_step, 24)
        self.assertEqual(server_config.guidance_scale, 2.5)
        self.assertFalse(server_config.denoise)
        self.assertEqual(server_config.position_temperature, 4.0)
        self.assertEqual(server_config.class_temperature, 0.2)
        self.assertEqual(server_config.layer_penalty_factor, 6.0)
        self.assertFalse(server_config.preprocess_voice_clone_prompt)
        self.assertFalse(server_config.postprocess_output_audio)
        self.assertEqual(server_config.audio_chunk_duration, 12.0)
        self.assertEqual(server_config.audio_chunk_threshold, 24.0)


if __name__ == "__main__":
    unittest.main()
