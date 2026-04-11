"""Model alias helpers for the supported faster-qwen3-tts voice-clone models."""

from __future__ import annotations

from pathlib import Path

MODEL_ALIASES = {
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}


def resolve_model_id(model_value: str) -> str:
    """Resolve a CLI or environment model value into a concrete model identifier.

    Usage:
        Accepts the short aliases required by this project (`0.6B` and `1.7B`),
        a full Hugging Face model ID, or a local filesystem path to an already
        downloaded model directory.

    Parameters:
        model_value: The raw model selector supplied by the user.

    Returns:
        The exact model identifier or local path string that should be passed to
        `FasterQwen3TTS.from_pretrained`.

    Raises:
        ValueError: If the supplied model value is empty or not one of the
        supported aliases / accepted direct identifiers.
    """
    normalized_value = model_value.strip()
    if not normalized_value:
        raise ValueError("Model selection cannot be empty.")

    if normalized_value in MODEL_ALIASES:
        return MODEL_ALIASES[normalized_value]

    if normalized_value.startswith("Qwen/"):
        return normalized_value

    if Path(normalized_value).expanduser().exists():
        return normalized_value

    raise ValueError(
        "Unsupported model selection "
        f"{model_value!r}. Use one of {sorted(MODEL_ALIASES)} or provide a full model ID/path."
    )


def expand_model_selection(model_value: str) -> list[str]:
    """Expand a prefetch-oriented model selector into one or more concrete model IDs.

    Usage:
        This helper is used by the model prefetch command so the caller can ask
        for a single supported model or all supported built-in aliases at once.

    Parameters:
        model_value: Either a regular model selector accepted by
        `resolve_model_id` or the special value `all`.

    Returns:
        A list of concrete model identifiers that should be downloaded.

    Raises:
        ValueError: If the input cannot be resolved into supported model IDs.
    """
    if model_value.strip().lower() == "all":
        return [MODEL_ALIASES[alias] for alias in sorted(MODEL_ALIASES)]

    return [resolve_model_id(model_value)]
