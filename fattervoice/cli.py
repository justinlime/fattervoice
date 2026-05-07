"""Command-line entry point for starting the combined fattervoice server."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Sequence

from .config import ServerConfig, format_server_config_summary, parse_server_config
from .runtime import configure_logging, run_server

LOGGER = logging.getLogger(__name__)



def emit_startup_config_summary(config: ServerConfig) -> None:
    """Log the resolved startup configuration at a level guaranteed to stay visible.

    Usage:
        The CLI calls this immediately after logging is configured so operators
        always see one boxed summary of the effective runtime settings, even when
        they choose a higher minimum log threshold such as WARNING or ERROR.

    Parameters:
        config: The fully resolved runtime configuration that should be displayed.

    Returns:
        None. The formatted startup summary is emitted through the module logger.
    """
    summary_log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    LOGGER.log(summary_log_level, "%s", format_server_config_summary(config))



def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse configuration, log the effective settings, and run the production server.

    Usage:
        This function is exposed as the `fattervoice` console script and can also
        be called programmatically by tests or alternative launch wrappers. After
        logging is configured, it emits one boxed summary of the resolved runtime
        configuration before handing control to the async server runtime.

    Parameters:
        argv: Optional explicit command-line arguments. When omitted, argparse
            reads the process command line.

    Returns:
        None. The function blocks until the server exits.
    """
    config = parse_server_config(argv)
    configure_logging(config.log_level)
    emit_startup_config_summary(config)

    try:
        asyncio.run(run_server(config))
    except KeyboardInterrupt:
        pass
