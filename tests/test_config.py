"""Unit tests for runtime configuration parsing."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fattervoice.config import format_server_config_summary, parse_server_config


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

    def test_parse_server_config_defaults_num_step_and_dtype(self) -> None:
        """Ensure runtime defaults now use 32 steps and bfloat16 precision.

        Usage:
            The production server intentionally shifted its default quality/speed
            tradeoff to 32 OmniVoice steps and changed its default precision to
            bfloat16. This regression test verifies both defaults whenever
            operators do not set CLI flags or environment overrides.

        Parameters:
            None.

        Returns:
            None. The test asserts that the parsed config uses the new defaults.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            voices_dir = Path(temp_dir) / "voices"
            voices_dir.mkdir()

            with patch.dict(os.environ, {}, clear=True):
                server_config = parse_server_config([
                    "--voices-dir",
                    str(voices_dir),
                ])

        self.assertEqual(server_config.num_step, 32)
        self.assertEqual(server_config.dtype, "bfloat16")

    def test_parse_server_config_uses_renamed_openapi_and_wyoming_environment_fallbacks(self) -> None:
        """Ensure renamed host/port environment variables feed the runtime config.

        Usage:
            The server configuration contract now separates OpenAI-compatible HTTP
            bind settings from Wyoming bind settings with renamed environment
            variables. This test verifies those env fallbacks populate the parsed
            config when CLI flags are omitted.

        Parameters:
            None.

        Returns:
            None. The test asserts that the parsed config picks up the renamed
            environment variables exactly.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            voices_dir = Path(temp_dir) / "voices"
            voices_dir.mkdir()

            with patch.dict(
                os.environ,
                {
                    "FATTERVOICE_OPENAPI_HOST": "127.0.0.1",
                    "FATTERVOICE_OPENAPI_PORT": "18000",
                    "FATTERVOICE_WYOMING_HOST": "192.168.1.50",
                    "FATTERVOICE_WYOMING_PORT": "11300",
                },
                clear=True,
            ):
                server_config = parse_server_config([
                    "--voices-dir",
                    str(voices_dir),
                ])

        self.assertEqual(server_config.openapi_host, "127.0.0.1")
        self.assertEqual(server_config.openapi_port, 18000)
        self.assertEqual(server_config.wyoming_host, "192.168.1.50")
        self.assertEqual(server_config.wyoming_port, 11300)
        self.assertEqual(server_config.wyoming_uri, "tcp://192.168.1.50:11300")

    def test_format_server_config_summary_groups_all_effective_settings(self) -> None:
        """Ensure startup summary renders the resolved config in a readable box.

        Usage:
            The CLI now logs a boxed configuration summary during startup. This
            regression test verifies that the formatter includes representative
            values from the network, runtime, and generation sections so
            operators can audit the effective runtime settings at a glance.

        Parameters:
            None.

        Returns:
            None. The test asserts that the formatted summary contains the box
            framing plus representative values from every major config section.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            voices_dir = Path(temp_dir) / "voices"
            voices_dir.mkdir()

            server_config = parse_server_config([
                "--voices-dir",
                str(voices_dir),
                "--openapi-host",
                "127.0.0.1",
                "--openapi-port",
                "9000",
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--default-language",
                "en",
                "--wyoming-host",
                "127.0.0.2",
                "--wyoming-port",
                "12000",
                "--log-level",
                "debug",
                "--num-step",
                "24",
                "--guidance-scale",
                "2.5",
                "--no-denoise",
                "--t-shift",
                "0.25",
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

        summary = format_server_config_summary(server_config)

        self.assertTrue(summary.startswith("╭"))
        self.assertTrue(summary.rstrip().endswith("╯"))
        self.assertIn("fattervoice startup configuration", summary)
        self.assertIn("Network", summary)
        self.assertIn("Runtime", summary)
        self.assertIn("Generation", summary)
        self.assertIn(str(voices_dir.resolve()), summary)
        self.assertIn("127.0.0.1", summary)
        self.assertIn("9000", summary)
        self.assertIn("cpu", summary)
        self.assertIn("float32", summary)
        self.assertIn("en", summary)
        self.assertNotIn("Max request text length", summary)
        self.assertNotIn("Wyoming URI", summary)
        self.assertNotIn("tcp://127.0.0.2:12000", summary)
        self.assertIn("DEBUG", summary)
        self.assertIn("2.5", summary)
        self.assertIn("0.25", summary)
        self.assertIn("6.0", summary)
        self.assertIn("12.0", summary)
        self.assertIn("24.0", summary)
        self.assertIn("false", summary)

    def test_parse_server_config_formats_ipv6_wyoming_hosts_safely(self) -> None:
        """Ensure split Wyoming bind settings produce a valid IPv6-safe URI.

        Usage:
            The runtime now stores Wyoming host and port separately and derives a
            final URI string for `AsyncServer.from_uri(...)`. This regression test
            verifies that IPv6 literals are wrapped in brackets before the port is
            appended so the resulting URI remains parseable.

        Parameters:
            None.

        Returns:
            None. The test asserts that the derived Wyoming URI uses bracketed
            IPv6 formatting.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            voices_dir = Path(temp_dir) / "voices"
            voices_dir.mkdir()

            server_config = parse_server_config([
                "--voices-dir",
                str(voices_dir),
                "--wyoming-host",
                "::1",
                "--wyoming-port",
                "10300",
            ])

        self.assertEqual(server_config.wyoming_uri, "tcp://[::1]:10300")

    def test_parse_server_config_supports_renamed_network_and_generation_flags(self) -> None:
        """Ensure renamed bind flags and OmniVoice tuning flags still parse correctly.

        Usage:
            Operators now configure OpenAI-compatible HTTP and Wyoming bind
            settings through renamed split host/port flags. This test verifies
            those flags and representative OmniVoice generation knobs land in the
            parsed runtime config together.

        Parameters:
            None.

        Returns:
            None. The test asserts that the parsed config contains the requested
            bind and OmniVoice settings.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            voices_dir = Path(temp_dir) / "voices"
            voices_dir.mkdir()

            server_config = parse_server_config([
                "--voices-dir",
                str(voices_dir),
                "--openapi-host",
                "127.0.0.1",
                "--openapi-port",
                "9000",
                "--wyoming-host",
                "127.0.0.2",
                "--wyoming-port",
                "12000",
                "--dtype",
                "float32",
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

        self.assertEqual(server_config.openapi_host, "127.0.0.1")
        self.assertEqual(server_config.openapi_port, 9000)
        self.assertEqual(server_config.wyoming_host, "127.0.0.2")
        self.assertEqual(server_config.wyoming_port, 12000)
        self.assertEqual(server_config.dtype, "float32")
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

    def test_parse_server_config_rejects_removed_model_and_legacy_network_flags(self) -> None:
        """Ensure removed legacy model, request-limit, and bind flags are rejected.

        Usage:
            The runtime model is now hardcoded to OmniVoice, the old generic
            host/port plus Wyoming-URI flags were replaced with renamed split
            bind flags, and request-length limiting is no longer configurable.
            This regression test verifies that legacy invocations now fail fast
            instead of being silently accepted.

        Parameters:
            None.

        Returns:
            None. The test asserts that argparse raises `SystemExit` for each
            removed legacy flag.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            voices_dir = Path(temp_dir) / "voices"
            voices_dir.mkdir()

            removed_flag_sets = [
                ["--model", "omnivoice"],
                ["--host", "0.0.0.0"],
                ["--port", "8000"],
                ["--max-text-length", "512"],
                ["--wyoming-enabled"],
                ["--wyoming-uri", "tcp://0.0.0.0:10300"],
            ]
            for removed_flags in removed_flag_sets:
                with self.subTest(removed_flags=removed_flags):
                    with self.assertRaises(SystemExit):
                        parse_server_config([
                            "--voices-dir",
                            str(voices_dir),
                            *removed_flags,
                        ])

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
