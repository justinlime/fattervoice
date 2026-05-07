"""Unit tests for runtime configuration parsing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fattervoice.config import parse_server_config


class ServerConfigTests(unittest.TestCase):
    """Verify that runtime configuration honors the documented CLI and env contract."""

    def test_parse_server_config_rejects_removed_cache_and_manifest_flags(self) -> None:
        """Ensure removed cache-related CLI flags now fail fast during parsing.

        Usage:
            The wrapper no longer exposes cache-path or prefetch-manifest server
            options, so this regression test verifies that operators get an
            argparse failure instead of a silently ignored flag.

        Parameters:
            None.

        Returns:
            None. The test asserts that argparse raises `SystemExit` for each
            removed flag.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            voices_dir = Path(temp_dir) / "voices"
            voices_dir.mkdir()

            with self.assertRaises(SystemExit):
                parse_server_config([
                    "--voices-dir",
                    str(voices_dir),
                    "--model-cache-dir",
                    "/tmp/hf-cache",
                ])

            with self.assertRaises(SystemExit):
                parse_server_config([
                    "--voices-dir",
                    str(voices_dir),
                    "--prefetch-manifest",
                    "/tmp/prefetched-models.json",
                ])

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

    def test_parse_server_config_rejects_removed_warmup_flag(self) -> None:
        """Ensure the removed warmup flag is no longer accepted by the CLI.

        Usage:
            Warmup support has been removed entirely, so this regression test
            verifies that operators receive an argparse failure instead of a
            silently ignored or partially supported flag.

        Parameters:
            None.

        Returns:
            None. The test asserts that argparse raises `SystemExit`.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            voices_dir = Path(temp_dir) / "voices"
            voices_dir.mkdir()

            with self.assertRaises(SystemExit):
                parse_server_config([
                    "--voices-dir",
                    str(voices_dir),
                    "--warmup",
                ])

    def test_parse_server_config_rejects_removed_wyoming_chunk_flag(self) -> None:
        """Ensure the removed Wyoming chunk-size flag is no longer accepted.

        Usage:
            Wyoming audio chunk sizing is now hardcoded, so this regression test
            verifies that old deployments fail fast if they still pass the removed
            override flag.

        Parameters:
            None.

        Returns:
            None. The test asserts that argparse raises `SystemExit`.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            voices_dir = Path(temp_dir) / "voices"
            voices_dir.mkdir()

            with self.assertRaises(SystemExit):
                parse_server_config([
                    "--voices-dir",
                    str(voices_dir),
                    "--wyoming-audio-chunk-samples",
                    "2048",
                ])


if __name__ == "__main__":
    unittest.main()
