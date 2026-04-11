"""Helpers for keeping Hugging Face cache paths consistent across tools and runtime."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional



def configure_huggingface_cache(cache_dir: Optional[Path]) -> Optional[Path]:
    """Configure Hugging Face cache environment variables to point at one hub cache path.

    Usage:
        Call this helper before `snapshot_download(...)` or `from_pretrained(...)`
        so build-time prefetching and runtime model loading both resolve files from
        the same on-disk Hugging Face hub cache directory.

    Parameters:
        cache_dir: The explicit hub cache directory to use. When `None`, the
            function leaves the current process environment unchanged.

    Returns:
        The resolved cache directory path that was applied, or `None` when no
        explicit cache directory was provided.
    """
    if cache_dir is None:
        return None

    resolved_cache_dir = cache_dir.expanduser().resolve()
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_CACHE"] = str(resolved_cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(resolved_cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(resolved_cache_dir)
    os.environ["HF_HOME"] = str(resolved_cache_dir.parent)
    return resolved_cache_dir



def is_huggingface_offline_mode_enabled() -> bool:
    """Return whether Hugging Face libraries are currently configured for offline mode.

    Usage:
        Runtime model-loading code calls this helper to decide when it should use
        stricter local-only loading behavior instead of allowing implicit Hub
        lookups.

    Parameters:
        None.

    Returns:
        `True` when either `HF_HUB_OFFLINE` or `TRANSFORMERS_OFFLINE` is set to a
        truthy value, otherwise `False`.
    """
    truthy_values = {"1", "true", "yes", "on"}
    return (
        os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in truthy_values
        or os.environ.get("TRANSFORMERS_OFFLINE", "").strip().lower() in truthy_values
    )
