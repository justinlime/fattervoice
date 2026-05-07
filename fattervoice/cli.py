"""Command-line entry point for starting the combined fattervoice server."""

from __future__ import annotations

import asyncio
from typing import Optional, Sequence

from .config import parse_server_config
from .runtime import configure_logging, run_server



def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse configuration, configure logging, and run the production server.

    Usage:
        This function is exposed as the `fattervoice` console script and can also
        be called programmatically by tests or alternative launch wrappers.

    Parameters:
        argv: Optional explicit command-line arguments. When omitted, argparse
            reads the process command line.

    Returns:
        None. The function blocks until the server exits.
    """
    config = parse_server_config(argv)
    configure_logging(config.log_level)

    try:
        asyncio.run(run_server(config))
    except KeyboardInterrupt:
        pass
