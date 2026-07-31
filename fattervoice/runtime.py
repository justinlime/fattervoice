"""Runtime orchestration for the shared model service, HTTP API, and Wyoming server."""

from __future__ import annotations

import asyncio
import logging
import sys
import uvicorn

from .config import ServerConfig
from .openai_api import create_openai_app
from .service import TtsService
from .voice_registry import VoiceRegistry
from .wyoming_server import run_wyoming_server


# ANSI color codes
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[91m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_BLUE = "\033[94m"
_MAGENTA = "\033[95m"
_CYAN = "\033[96m"
_WHITE = "\033[97m"
_GRAY = "\033[90m"

# Level colors
_LEVEL_COLORS = {
    logging.DEBUG: _CYAN,
    logging.INFO: _GREEN,
    logging.WARNING: _YELLOW,
    logging.ERROR: _RED,
    logging.CRITICAL: _RED + _BOLD,
}

# Module name colors (short name -> color)
_MODULE_COLORS = {
    "service": _MAGENTA,
    "openai_api": _BLUE,
    "wyoming_server": _YELLOW,
    "voice_registry": _WHITE,
    "config": _GRAY,
}


class _ColoredFormatter(logging.Formatter):
    """Logging formatter that colors output by level and module.

    Colors are only applied when stdout is a TTY; they are stripped
    automatically when logs are piped to a file or journal.
    """

    def __init__(self, use_color: bool = True) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        self._use_color = use_color and sys.stdout.isatty()

    def _color_for_level(self, levelno: int) -> str:
        if not self._use_color:
            return ""
        return _LEVEL_COLORS.get(levelno, "")

    def _color_for_name(self, name: str) -> str:
        if not self._use_color:
            return ""
        # Use the last part of the dotted name (e.g. "service" from "fattervoice.service")
        short = name.rsplit(".", 1)[-1] if "." in name else name
        return _MODULE_COLORS.get(short, "")

    def format(self, record: logging.LogRecord) -> str:
        level_color = self._color_for_level(record.levelno)
        name_color = self._color_for_name(record.name)

        # Color the level name
        record.levelname = f"{level_color}{_BOLD}{record.levelname}{_RESET}"

        # Color the module name in brackets
        record.name = f"{name_color}{record.name}{_RESET}"

        # Dim the timestamp slightly
        formatted = super().format(record)
        if self._use_color:
            # Replace the timestamp at the start with a dimmed version
            # Format: "HH:MM:SS LEVEL [name] message"
            parts = formatted.split(" ", 2)
            if len(parts) >= 3:
                formatted = f"{_DIM}{parts[0]}{_RESET} {parts[1]} {parts[2]}"

        return formatted


def configure_logging(log_level: str) -> None:
    """Configure process-wide logging for the server entry point.

    Usage:
        The CLI calls this once before any long-lived services start so startup,
        HTTP, and Wyoming logs all share the same minimum severity threshold.

        Logs are colorized when stdout is a TTY and fall back to plain text
        when piped to a file or journal.

    Parameters:
        log_level: The textual Python logging level such as `INFO` or `DEBUG`.

    Returns:
        None. The global logging system is configured in place.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_ColoredFormatter())
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        handlers=[handler],
    )


async def run_http_server(service: TtsService, voice_registry: VoiceRegistry, config: ServerConfig) -> None:
    """Start the FastAPI/Uvicorn HTTP server and block while it handles requests.

    Usage:
        The main runtime starts this coroutine after the shared synthesis service
        is ready so the OpenAI-compatible API only accepts traffic once the model
        and voice registry are available.

    Parameters:
        service: Shared synthesis service used by all HTTP requests.
        voice_registry: Shared validated voice registry exposed via HTTP discovery routes.
        config: Immutable runtime configuration containing the OpenAI-compatible HTTP bind host and port.

    Returns:
        None. The coroutine runs until the HTTP server shuts down.
    """
    application = create_openai_app(service=service, voice_registry=voice_registry, config=config)
    uvicorn_config = uvicorn.Config(
        application,
        host=config.openapi_host,
        port=config.openapi_port,
        log_level=config.log_level.lower(),
    )
    await uvicorn.Server(uvicorn_config).serve()


async def run_server(config: ServerConfig) -> None:
    """Load shared state and run the HTTP and Wyoming servers together.

    Usage:
        The CLI entry point calls this coroutine after parsing configuration. It
        ensures both protocol adapters share one validated voice registry and one
        long-lived model instance.

    Parameters:
        config: Immutable runtime configuration for the entire process.

    Returns:
        None. The coroutine runs until one of the server tasks exits or the
        process is interrupted.
    """
    voice_registry = VoiceRegistry.scan(config.voices_dir)
    service = TtsService(config=config, voice_registry=voice_registry)
    await service.start()

    try:
        server_tasks = {
            asyncio.create_task(
                run_http_server(service=service, voice_registry=voice_registry, config=config),
                name="fattervoice-http",
            ),
            asyncio.create_task(
                run_wyoming_server(
                    service=service,
                    voice_registry=voice_registry,
                    config=config,
                ),
                name="fattervoice-wyoming",
            ),
        }

        done_tasks, pending_tasks = await asyncio.wait(
            server_tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )

        for completed_task in done_tasks:
            completed_task.result()

        for pending_task in pending_tasks:
            pending_task.cancel()

        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
    finally:
        service.close()
