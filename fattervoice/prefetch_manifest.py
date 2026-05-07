"""Helpers for resolving prefetched Hugging Face model snapshots from the local cache."""

from __future__ import annotations

from pathlib import Path
from typing import Optional



def resolve_snapshot_directory_from_cached_file(cached_file_path: str) -> Path:
    """Derive the containing snapshot directory from one cached Hugging Face file path.

    Usage:
        `try_to_load_from_cache(...)` returns a cached file path, which in the
        standard Hugging Face cache layout usually lives under
        `snapshots/<revision>/...` and may itself be a symlink into `blobs/`.
        This helper intentionally takes the parent directory before resolving the
        path so the snapshot directory is preserved instead of collapsing into
        the shared blob store.

    Parameters:
        cached_file_path: The cached file path returned by Hugging Face cache
            lookup helpers.

    Returns:
        The resolved snapshot directory that contains the cached file path.
    """
    return Path(cached_file_path).expanduser().parent.resolve()



def resolve_cached_model_snapshot_path(model_id: str, cache_dir: Optional[Path]) -> str:
    """Resolve a model ID to a cached local snapshot path using Hugging Face cache helpers.

    Usage:
        Runtime startup calls this helper to find an already-downloaded local
        snapshot directory without making network requests. The wrapper now uses
        cache inspection only and no longer depends on a separate prefetch
        manifest file.

    Parameters:
        model_id: The concrete Hugging Face model ID or local filesystem path
            that should be resolved.
        cache_dir: The optional Hugging Face hub cache directory to inspect.
            When `None`, the active Hugging Face default cache configuration is
            used.

    Returns:
        The resolved local snapshot directory when one can be located in the
        cache or when `model_id` already points at an existing local path;
        otherwise the original `model_id` string.
    """
    candidate_path = Path(model_id).expanduser()
    if candidate_path.exists():
        return str(candidate_path.resolve())

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
        cached_snapshot_path = resolve_snapshot_directory_from_cached_file(cached_config_path)
        if cached_snapshot_path.name != "blobs" and cached_snapshot_path.exists():
            return str(cached_snapshot_path)

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
