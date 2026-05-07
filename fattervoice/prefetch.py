"""Offline model prefetch helper for Docker builds and manual cache warming."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from huggingface_hub import snapshot_download

from .model_catalog import expand_model_selection, expand_prefetch_asset_ids



def prefetch_models(model_ids: Sequence[str]) -> dict[str, Path]:
    """Download one or more Hugging Face model snapshots into the active cache.

    Usage:
        Docker builds and manual operators call this helper to ensure required
        OmniVoice artifacts are available before the runtime enters an offline
        environment. The helper now relies on the process's current Hugging Face
        cache configuration instead of accepting a wrapper-specific cache path.

    Parameters:
        model_ids: Concrete model IDs or local filesystem paths.

    Returns:
        A mapping from each requested model ID to the resolved local snapshot
        path that should be used for offline loading.
    """
    prefetched_model_paths: dict[str, Path] = {}

    for model_id in model_ids:
        candidate_path = Path(model_id).expanduser()
        if candidate_path.exists():
            prefetched_model_paths[model_id] = candidate_path.resolve()
            continue

        snapshot_path = snapshot_download(repo_id=model_id)
        prefetched_model_paths[model_id] = Path(snapshot_path)

    return prefetched_model_paths



def build_prefetch_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser for the offline model prefetch command.

    Usage:
        The `fattervoice-prefetch` console script uses this parser to accept a
        model alias or `all`. Cache-location behavior is intentionally delegated
        to the active Hugging Face environment instead of wrapper-specific flags.

    Parameters:
        None.

    Returns:
        A configured `argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(description="Prefetch OmniVoice model artifacts into the Hugging Face cache.")
    parser.add_argument(
        "--model",
        default="omnivoice",
        help="Model alias, direct model ID/path, or `all`.",
    )
    return parser



def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse CLI arguments and prefetch the requested model snapshots.

    Usage:
        This function is exposed as the `fattervoice-prefetch` console script and
        is especially useful during Docker image builds.

    Parameters:
        argv: Optional explicit command-line arguments. When omitted, argparse
            reads the process command line.

    Returns:
        None. The function prints the local paths of the selected runtime models
        and any required auxiliary OmniVoice assets.
    """
    parser = build_prefetch_parser()
    args = parser.parse_args(argv)

    requested_model_ids = expand_model_selection(args.model)
    prefetched_model_paths = prefetch_models(expand_prefetch_asset_ids(args.model))

    for model_id in requested_model_ids:
        print(prefetched_model_paths[model_id])


if __name__ == "__main__":
    main()
