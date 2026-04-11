"""Helpers for recording and resolving exact local paths for prefetched models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional



def write_prefetch_manifest(model_paths: dict[str, Path], manifest_path: Path) -> Path:
    """Write a manifest that maps remote model IDs to local prefetched snapshot paths.

    Usage:
        Docker builds and manual prefetch commands call this helper after models
        have been downloaded so runtime startup can resolve an exact local model
        directory instead of relying only on Hugging Face cache discovery.

    Parameters:
        model_paths: A mapping from concrete model IDs such as
            `Qwen/Qwen3-TTS-12Hz-1.7B-Base` to the local snapshot directories
            returned by `snapshot_download(...)`.
        manifest_path: The JSON file path that should receive the manifest.

    Returns:
        The resolved manifest path that was written to disk.
    """
    resolved_manifest_path = manifest_path.expanduser().resolve()
    resolved_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    serialized_manifest = {
        "model_paths": {
            model_id: str(model_path.expanduser().resolve())
            for model_id, model_path in sorted(model_paths.items())
        }
    }
    resolved_manifest_path.write_text(
        json.dumps(serialized_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return resolved_manifest_path



def read_prefetch_manifest(manifest_path: Optional[Path]) -> dict[str, str]:
    """Load a prefetched-model manifest from disk if one is available.

    Usage:
        Runtime model resolution calls this helper before startup so offline
        containers can map remote model IDs onto already-downloaded snapshot paths.

    Parameters:
        manifest_path: The optional JSON file path to read.

    Returns:
        A mapping from concrete model IDs to local snapshot path strings. When the
        file is missing or no path is provided, an empty mapping is returned.
    """
    if manifest_path is None:
        return {}

    resolved_manifest_path = manifest_path.expanduser().resolve()
    if not resolved_manifest_path.exists():
        return {}

    manifest_data = json.loads(resolved_manifest_path.read_text(encoding="utf-8"))
    model_paths = manifest_data.get("model_paths", {})
    if not isinstance(model_paths, dict):
        return {}

    return {
        str(model_id): str(model_path)
        for model_id, model_path in model_paths.items()
        if isinstance(model_id, str) and isinstance(model_path, str)
    }



def resolve_cached_model_snapshot_path(model_id: str, cache_dir: Optional[Path]) -> str:
    """Resolve a model ID to a cached local snapshot path using Hugging Face cache helpers.

    Usage:
        Runtime startup calls this helper as a fallback when no explicit manifest
        entry is available. It uses Hugging Face cache inspection utilities that
        do not perform network requests.

    Parameters:
        model_id: The concrete Hugging Face model ID that should be resolved.
        cache_dir: The optional Hugging Face hub cache directory to inspect.

    Returns:
        The resolved local snapshot directory when one can be located in the
        cache; otherwise the original `model_id` string.
    """
    resolved_cache_dir = cache_dir.expanduser().resolve() if cache_dir is not None else None

    try:
        from huggingface_hub import scan_cache_dir, try_to_load_from_cache
    except ImportError:
        return model_id

    cached_config_path = try_to_load_from_cache(
        repo_id=model_id,
        filename="config.json",
        cache_dir=str(resolved_cache_dir) if resolved_cache_dir is not None else None,
        revision="main",
        repo_type="model",
    )
    if isinstance(cached_config_path, str):
        return str(Path(cached_config_path).expanduser().resolve().parent)

    try:
        cache_info = scan_cache_dir(str(resolved_cache_dir) if resolved_cache_dir is not None else None)
    except Exception:
        return model_id

    matching_repos = [
        cached_repo
        for cached_repo in cache_info.repos
        if cached_repo.repo_id == model_id and cached_repo.repo_type == "model"
    ]
    if not matching_repos:
        return model_id

    matching_repo = max(matching_repos, key=lambda cached_repo: cached_repo.last_modified)
    main_revision = next(
        (
            cached_revision
            for cached_revision in matching_repo.revisions
            if "main" in cached_revision.refs and cached_revision.snapshot_path.exists()
        ),
        None,
    )
    if main_revision is not None:
        return str(main_revision.snapshot_path.expanduser().resolve())

    existing_revisions = [
        cached_revision
        for cached_revision in matching_repo.revisions
        if cached_revision.snapshot_path.exists()
    ]
    if not existing_revisions:
        return model_id

    latest_revision = max(existing_revisions, key=lambda cached_revision: cached_revision.last_modified)
    return str(latest_revision.snapshot_path.expanduser().resolve())



def resolve_prefetched_model_path(
    model_id: str,
    manifest_path: Optional[Path],
    cache_dir: Optional[Path],
) -> str:
    """Resolve a model ID to a local snapshot path using manifest data or cache inspection.

    Usage:
        Server startup calls this helper after normal alias resolution so offline
        containers can load an exact local snapshot directory recorded during the
        image build, while still having a robust fallback to Hugging Face cache
        inspection if the manifest is missing or stale.

    Parameters:
        model_id: The concrete model ID or local path selected for runtime use.
        manifest_path: The optional manifest path written during prefetch.
        cache_dir: The optional Hugging Face hub cache directory to inspect when
            the manifest does not provide a usable path.

    Returns:
        The local snapshot path when one can be resolved locally; otherwise the
        original `model_id` string.
    """
    prefetched_model_paths = read_prefetch_manifest(manifest_path)
    prefetched_path = prefetched_model_paths.get(model_id)
    if prefetched_path is not None:
        resolved_prefetched_path = Path(prefetched_path).expanduser().resolve()
        if resolved_prefetched_path.exists():
            return str(resolved_prefetched_path)

    return resolve_cached_model_snapshot_path(model_id, cache_dir)
