"""Unit tests for runtime configuration parsing."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fatterqwen.config import parse_server_config


class ServerConfigTests(unittest.TestCase):
    """Verify that runtime configuration honors the documented CLI and env contract."""

    def test_parse_server_config_uses_huggingface_cache_environment_fallbacks(self) -> None:
        """Ensure runtime startup can reuse the Docker image's Hugging Face cache env vars.

        Usage:
            The Docker image sets standard Hugging Face cache environment variables
            even when `FATTERQWEN_MODEL_CACHE_DIR` is not explicitly provided.
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
                    "FATTERQWEN_MODEL_CACHE_DIR": "",
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

    def test_parse_server_config_supports_quality_pipeline_flags(self) -> None:
        """Ensure the new voice-quality options can be configured through the CLI.

        Usage:
            The quality pipeline introduced for reference preprocessing and
            long-form chunking adds several new runtime knobs. This test verifies
            that representative CLI flags land in the parsed server config.

        Parameters:
            None.

        Returns:
            None. The test asserts that the parsed config contains the requested
            quality-pipeline settings.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            voices_dir = Path(temp_dir) / "voices"
            voices_dir.mkdir()

            server_config = parse_server_config([
                "--voices-dir",
                str(voices_dir),
                "--no-preprocess-reference-audio",
                "--no-normalize-reference-transcript",
                "--reference-prompt-target-rms",
                "0.2",
                "--no-postprocess-output-audio",
                "--longform-chunk-threshold-units",
                "500",
                "--longform-target-units",
                "300",
                "--longform-min-units",
                "90",
                "--longform-crossfade-milliseconds",
                "40",
                "--longform-gap-milliseconds",
                "70",
            ])

        self.assertFalse(server_config.preprocess_reference_audio)
        self.assertFalse(server_config.normalize_reference_transcript)
        self.assertEqual(server_config.reference_prompt_target_rms, 0.2)
        self.assertFalse(server_config.postprocess_output_audio)
        self.assertEqual(server_config.longform_chunk_threshold_units, 500)
        self.assertEqual(server_config.longform_target_units, 300)
        self.assertEqual(server_config.longform_min_units, 90)
        self.assertEqual(server_config.longform_crossfade_milliseconds, 40)
        self.assertEqual(server_config.longform_gap_milliseconds, 70)


if __name__ == "__main__":
    unittest.main()
