"""Offline model prefetch helper for Docker builds and manual cache warming."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence

from huggingface_hub import snapshot_download

from .hf_cache import configure_huggingface_cache
from .model_catalog import expand_model_selection, expand_prefetch_asset_ids
from .prefetch_manifest import write_prefetch_manifest



def prefetch_models(model_ids: Sequence[str], cache_dir: Optional[Path]) -> dict[str, Path]:
    """Download one or more Hugging Face model snapshots into the local cache.

    Usage:
        Docker builds and manual operators call this helper to ensure required
        OmniVoice artifacts are available before the runtime enters an offline
        environment.

    Parameters:
        model_ids: Concrete model IDs or local filesystem paths.
        cache_dir: Optional Hugging Face hub cache directory to populate.

    Returns:
        A mapping from each requested model ID to the resolved local snapshot
        path that should be used for offline loading.
    """
    prefetched_model_paths: dict[str, Path] = {}
    resolved_cache_dir = configure_huggingface_cache(cache_dir)

    for model_id in model_ids:
        candidate_path = Path(model_id).expanduser()
        if candidate_path.exists():
            prefetched_model_paths[model_id] = candidate_path.resolve()
            continue

        snapshot_path = snapshot_download(
            repo_id=model_id,
            cache_dir=str(resolved_cache_dir) if resolved_cache_dir else None,
        )
        prefetched_model_paths[model_id] = Path(snapshot_path)

    return prefetched_model_paths



def build_prefetch_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser for the offline model prefetch command.

    Usage:
        The `fattervoice-prefetch` console script uses this parser to accept a
        model alias or `all`, an optional cache directory, and an optional path
        where the resulting local snapshot manifest should be written.

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
    parser.add_argument(
        "--cache-dir",
        default=(
            os.environ.get("FATTERVOICE_MODEL_CACHE_DIR")
            or os.environ.get("HF_HUB_CACHE")
            or os.environ.get("HUGGINGFACE_HUB_CACHE")
            or os.environ.get("TRANSFORMERS_CACHE")
            or ""
        ),
        help="Optional Hugging Face hub cache directory to populate.",
    )
    parser.add_argument(
        "--manifest",
        default=os.environ.get("FATTERVOICE_PREFETCH_MANIFEST", ""),
        help="Optional JSON manifest path that should record local snapshot locations.",
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
        and optionally writes a manifest that records exact local cache locations
        for both the primary model and any required auxiliary OmniVoice assets.
    """
    parser = build_prefetch_parser()
    args = parser.parse_args(argv)

    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else None
    requested_model_ids = expand_model_selection(args.model)
    prefetched_model_paths = prefetch_models(
        expand_prefetch_asset_ids(args.model),
        cache_dir,
    )

    if args.manifest:
        write_prefetch_manifest(
            prefetched_model_paths,
            Path(args.manifest).expanduser().resolve(),
        )

    for model_id in requested_model_ids:
        print(prefetched_model_paths[model_id])


if __name__ == "__main__":
    main()
