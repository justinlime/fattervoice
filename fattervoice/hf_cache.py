"""Helpers for Hugging Face runtime environment inspection."""

from __future__ import annotations

import os



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
