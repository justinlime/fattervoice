"""Unit tests for CLI startup behavior."""

from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fattervoice.cli import main
from fattervoice.config import ServerConfig


class CliTests(unittest.TestCase):
    """Verify the CLI startup flow logs the resolved runtime configuration."""

    def test_main_logs_boxed_configuration_summary_before_running_server(self) -> None:
        """Ensure the CLI emits the startup summary before launching async runtime.

        Usage:
            Operators rely on the startup summary to verify the effective
            configuration. This test patches the async runtime and logging calls
            so it can assert that `main(...)` logs the boxed summary without
            starting real network services.

        Parameters:
            None.

        Returns:
            None. The test asserts that logging and runtime orchestration receive
            the expected resolved configuration.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            voices_dir = Path(temp_dir) / "voices"
            voices_dir.mkdir()
            server_config = ServerConfig(
                voices_dir=voices_dir.resolve(),
                openapi_host="0.0.0.0",
                openapi_port=8000,
                device="cpu",
                dtype="float32",
                default_language="auto",
                wyoming_host="0.0.0.0",
                wyoming_port=10300,
                log_level="ERROR",
            )

            with patch("fattervoice.cli.parse_server_config", return_value=server_config) as parse_mock, patch(
                "fattervoice.cli.configure_logging"
            ) as configure_logging_mock, patch(
                "fattervoice.cli.format_server_config_summary", return_value="BOXED CONFIG"
            ) as format_mock, patch(
                "fattervoice.cli.run_server", new_callable=AsyncMock
            ) as run_server_mock, patch(
                "fattervoice.cli.asyncio.run",
                side_effect=lambda coroutine: coroutine.close(),
            ) as asyncio_run_mock, patch("fattervoice.cli.LOGGER") as logger_mock:
                main(["--voices-dir", str(voices_dir)])

        parse_mock.assert_called_once_with(["--voices-dir", str(voices_dir)])
        configure_logging_mock.assert_called_once_with("ERROR")
        format_mock.assert_called_once_with(server_config)
        logger_mock.log.assert_called_once_with(logging.ERROR, "%s", "BOXED CONFIG")
        run_server_mock.assert_called_once_with(server_config)
        asyncio_run_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
