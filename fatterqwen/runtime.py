"""Runtime orchestration for the shared model service, HTTP API, and Wyoming server."""

from __future__ import annotations

import asyncio
import logging

import uvicorn

from .config import ServerConfig
from .openai_api import create_openai_app
from .service import TtsService
from .voice_registry import VoiceRegistry
from .wyoming_server import run_wyoming_server



def configure_logging(log_level: str) -> None:
    """Configure process-wide logging for the server entry point.

    Usage:
        The CLI calls this once before any long-lived services start so startup,
        HTTP, and Wyoming logs all share the same minimum severity threshold.

    Parameters:
        log_level: The textual Python logging level such as `INFO` or `DEBUG`.

    Returns:
        None. The global logging system is configured in place.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
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
        config: Immutable runtime configuration containing the bind host and port.

    Returns:
        None. The coroutine runs until the HTTP server shuts down.
    """
    application = create_openai_app(service=service, voice_registry=voice_registry, config=config)
    uvicorn_config = uvicorn.Config(
        application,
        host=config.host,
        port=config.port,
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
                name="fatterqwen-http",
            )
        }
        if config.wyoming_enabled:
            server_tasks.add(
                asyncio.create_task(
                    run_wyoming_server(
                        service=service,
                        voice_registry=voice_registry,
                        config=config,
                    ),
                    name="fatterqwen-wyoming",
                )
            )

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
