"""Model alias helpers for the supported OmniVoice runtime models."""

from __future__ import annotations

from pathlib import Path

_OMNIVOICE_MODEL_ID = "k2-fsa/OmniVoice"

MODEL_ALIASES = {
    "default": _OMNIVOICE_MODEL_ID,
    "omnivoice": _OMNIVOICE_MODEL_ID,
}

MODEL_PREFETCH_DEPENDENCIES = {
    _OMNIVOICE_MODEL_ID: ["eustlb/higgs-audio-v2-tokenizer"],
}


def resolve_model_id(model_value: str) -> str:
    """Resolve a CLI or environment model value into a concrete model identifier.

    Usage:
        Accepts the built-in OmniVoice aliases used by this project, a full
        Hugging Face model ID, or a local filesystem path to an already
        downloaded model directory.

    Parameters:
        model_value: The raw model selector supplied by the user.

    Returns:
        The exact model identifier or local path string that should be passed to
        `OmniVoice.from_pretrained`.

    Raises:
        ValueError: If the supplied model value is empty or not one of the
        supported aliases / accepted direct identifiers.
    """
    normalized_value = model_value.strip()
    if not normalized_value:
        raise ValueError("Model selection cannot be empty.")

    normalized_key = normalized_value.lower()
    if normalized_key in MODEL_ALIASES:
        return MODEL_ALIASES[normalized_key]

    if Path(normalized_value).expanduser().exists():
        return normalized_value

    if "/" in normalized_value:
        return normalized_value

    raise ValueError(
        "Unsupported model selection "
        f"{model_value!r}. Use one of {sorted(MODEL_ALIASES)} or provide a full model ID/path."
    )



def expand_model_selection(model_value: str) -> list[str]:
    """Expand a prefetch-oriented model selector into one or more concrete model IDs.

    Usage:
        This helper is used by the model prefetch command so the caller can ask
        for the built-in OmniVoice model or all supported built-in aliases at
        once.

    Parameters:
        model_value: Either a regular model selector accepted by
            `resolve_model_id` or the special value `all`.

    Returns:
        A list of concrete model identifiers that should be downloaded.

    Raises:
        ValueError: If the input cannot be resolved into supported model IDs.
    """
    if model_value.strip().lower() == "all":
        return sorted(set(MODEL_ALIASES.values()))

    return [resolve_model_id(model_value)]



def expand_prefetch_asset_ids(model_value: str) -> list[str]:
    """Expand a model selector into the full set of Hugging Face repos to prefetch.

    Usage:
        OmniVoice inference depends on the main model snapshot and may also need
        auxiliary repositories such as the Higgs audio tokenizer. Docker builds
        and manual cache-warming commands call this helper so offline runtime has
        every required repository available locally.

    Parameters:
        model_value: Either a regular model selector accepted by
            `resolve_model_id` or the special value `all`.

    Returns:
        A de-duplicated ordered list of Hugging Face model IDs that should be
        prefetched for the selected runtime model or models.
    """
    selected_model_ids = expand_model_selection(model_value)
    ordered_prefetch_ids: list[str] = []

    for model_id in selected_model_ids:
        if model_id not in ordered_prefetch_ids:
            ordered_prefetch_ids.append(model_id)

        for dependency_model_id in MODEL_PREFETCH_DEPENDENCIES.get(model_id, []):
            if dependency_model_id not in ordered_prefetch_ids:
                ordered_prefetch_ids.append(dependency_model_id)

    return ordered_prefetch_ids
