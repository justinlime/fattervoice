"""CLI and environment-based configuration for the fatterqwen server."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


@dataclass(frozen=True)
class ServerConfig:
    """Immutable runtime configuration shared by the HTTP and Wyoming servers."""

    voices_dir: Path
    host: str
    port: int
    model: str
    device: str
    dtype: str
    default_language: str
    chunk_size: int
    append_silence: bool
    non_streaming_mode: bool
    max_text_length: int
    model_cache_dir: Optional[Path]
    prefetch_manifest_path: Optional[Path]
    warmup: bool
    warmup_text: str
    wyoming_enabled: bool
    wyoming_uri: Optional[str]
    wyoming_audio_chunk_samples: int
    log_level: str
    preprocess_reference_audio: bool = True
    normalize_reference_transcript: bool = True
    reference_prompt_target_rms: float = 0.1
    postprocess_output_audio: bool = True
    longform_chunking_enabled: bool = True
    longform_chunk_threshold_units: int = 320
    longform_target_units: int = 200
    longform_min_units: int = 70
    longform_crossfade_milliseconds: int = 80
    longform_gap_milliseconds: int = 120


def environment_default(name: str, default: str) -> str:
    """Return a string configuration value using CLI-first / ENV-fallback semantics.

    Usage:
        This helper keeps environment access centralized so every CLI flag can use
        the same resolution rule and remain easy to audit.

    Parameters:
        name: The environment variable name to read.
        default: The value to use when the environment variable is unset.

    Returns:
        The environment variable value when present; otherwise the provided default.
    """
    return os.environ.get(name, default)



def environment_flag(name: str, default: bool) -> bool:
    """Parse a boolean environment variable into a Python boolean.

    Usage:
        This helper accepts common truthy and falsy string values so deployment
        environments can control optional features without custom parsing code.

    Parameters:
        name: The environment variable name to read.
        default: The boolean value to use when the environment variable is unset.

    Returns:
        The parsed boolean value.

    Raises:
        ValueError: If the environment variable is set but cannot be interpreted
        as a boolean.
    """
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default

    normalized_value = raw_value.strip().lower()
    if normalized_value in {"1", "true", "yes", "on"}:
        return True
    if normalized_value in {"0", "false", "no", "off"}:
        return False

    raise ValueError(
        f"Environment variable {name} must be a boolean string, got {raw_value!r}."
    )



def build_argument_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser for the production TTS server.

    Usage:
        The resulting parser exposes all primary runtime settings required by the
        project brief, while still allowing environment variables to act as the
        fallback source of truth.

    Parameters:
        None.

    Returns:
        A fully configured `argparse.ArgumentParser` instance.
    """
    parser = argparse.ArgumentParser(
        description="Production-oriented OpenAI-compatible and Wyoming TTS server for faster-qwen3-tts.",
    )
    parser.add_argument(
        "--voices-dir",
        default=environment_default("FATTERQWEN_VOICES_DIR", "voices"),
        help="Directory containing <voice>.<audio> and <voice>.txt pairs.",
    )
    parser.add_argument(
        "--host",
        default=environment_default("FATTERQWEN_HOST", "0.0.0.0"),
        help="HTTP bind host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(environment_default("FATTERQWEN_PORT", "8000")),
        help="HTTP bind port.",
    )
    parser.add_argument(
        "--model",
        default=environment_default("FATTERQWEN_MODEL", "1.7B"),
        help="Model alias (`0.6B` or `1.7B`) or a direct model ID/path.",
    )
    parser.add_argument(
        "--device",
        default=environment_default("FATTERQWEN_DEVICE", "cuda"),
        help="Torch device passed through to faster-qwen3-tts.",
    )
    parser.add_argument(
        "--dtype",
        default=environment_default("FATTERQWEN_DTYPE", "bfloat16"),
        help="Torch dtype name such as bfloat16 or float16.",
    )
    parser.add_argument(
        "--default-language",
        default=environment_default("FATTERQWEN_DEFAULT_LANGUAGE", "Auto"),
        help="Fallback language passed to synthesis when the client does not specify one.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(environment_default("FATTERQWEN_CHUNK_SIZE", "8")),
        help="Streaming chunk size passed to faster-qwen3-tts.",
    )
    parser.add_argument(
        "--append-silence",
        action=argparse.BooleanOptionalAction,
        default=environment_flag("FATTERQWEN_APPEND_SILENCE", True),
        help="Append a small amount of silence to reference audio before prompt creation.",
    )
    parser.add_argument(
        "--non-streaming-mode",
        action=argparse.BooleanOptionalAction,
        default=environment_flag("FATTERQWEN_NON_STREAMING_MODE", False),
        help="Pass the upstream non_streaming_mode flag during generation.",
    )
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=int(environment_default("FATTERQWEN_MAX_TEXT_LENGTH", "4000")),
        help="Maximum request text length accepted by the server.",
    )
    parser.add_argument(
        "--preprocess-reference-audio",
        action=argparse.BooleanOptionalAction,
        default=environment_flag("FATTERQWEN_PREPROCESS_REFERENCE_AUDIO", True),
        help="Remove excessive silence from reference prompt audio before caching it for voice cloning.",
    )
    parser.add_argument(
        "--normalize-reference-transcript",
        action=argparse.BooleanOptionalAction,
        default=environment_flag("FATTERQWEN_NORMALIZE_REFERENCE_TRANSCRIPT", True),
        help="Add missing terminal punctuation to reference transcripts before synthesis.",
    )
    parser.add_argument(
        "--reference-prompt-target-rms",
        type=float,
        default=float(environment_default("FATTERQWEN_REFERENCE_PROMPT_TARGET_RMS", "0.1")),
        help="Minimum RMS level applied to very quiet reference audio during prompt preparation.",
    )
    parser.add_argument(
        "--postprocess-output-audio",
        action=argparse.BooleanOptionalAction,
        default=environment_flag("FATTERQWEN_POSTPROCESS_OUTPUT_AUDIO", True),
        help="Compact excessive silences in generated audio before returning it to clients.",
    )
    parser.add_argument(
        "--longform-chunking",
        action=argparse.BooleanOptionalAction,
        default=environment_flag("FATTERQWEN_LONGFORM_CHUNKING", True),
        help="Split large requests into multiple synthesis calls with smoothed chunk boundaries.",
    )
    parser.add_argument(
        "--longform-chunk-threshold-units",
        type=int,
        default=int(environment_default("FATTERQWEN_LONGFORM_CHUNK_THRESHOLD_UNITS", "320")),
        help="Estimated speech-unit threshold above which long-form chunking is activated.",
    )
    parser.add_argument(
        "--longform-target-units",
        type=int,
        default=int(environment_default("FATTERQWEN_LONGFORM_TARGET_UNITS", "200")),
        help="Preferred estimated speech-unit size for each long-form synthesis chunk.",
    )
    parser.add_argument(
        "--longform-min-units",
        type=int,
        default=int(environment_default("FATTERQWEN_LONGFORM_MIN_UNITS", "70")),
        help="Minimum estimated speech-unit size for a trailing chunk before it is merged back into the previous one.",
    )
    parser.add_argument(
        "--longform-crossfade-milliseconds",
        type=int,
        default=int(environment_default("FATTERQWEN_LONGFORM_CROSSFADE_MILLISECONDS", "80")),
        help="Fade duration used when smoothing the boundary between long-form synthesis chunks.",
    )
    parser.add_argument(
        "--longform-gap-milliseconds",
        type=int,
        default=int(environment_default("FATTERQWEN_LONGFORM_GAP_MILLISECONDS", "120")),
        help="Silence gap inserted between long-form synthesis chunks before the next chunk fades in.",
    )
    parser.add_argument(
        "--model-cache-dir",
        default=(
            environment_default("FATTERQWEN_MODEL_CACHE_DIR", "")
            or environment_default("HF_HUB_CACHE", "")
            or environment_default("HUGGINGFACE_HUB_CACHE", "")
            or environment_default("TRANSFORMERS_CACHE", "")
        ),
        help="Optional Hugging Face hub cache directory for model artifacts.",
    )
    parser.add_argument(
        "--prefetch-manifest",
        default=environment_default("FATTERQWEN_PREFETCH_MANIFEST", ""),
        help="Optional JSON manifest that maps prefetched model IDs to exact local snapshot paths.",
    )
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=environment_flag("FATTERQWEN_WARMUP", False),
        help="Run a short warmup generation during startup to capture CUDA graphs before traffic arrives.",
    )
    parser.add_argument(
        "--warmup-text",
        default=environment_default("FATTERQWEN_WARMUP_TEXT", "Hello from fatterqwen."),
        help="Text used when warmup is enabled.",
    )
    parser.add_argument(
        "--wyoming-enabled",
        action=argparse.BooleanOptionalAction,
        default=environment_flag("FATTERQWEN_WYOMING_ENABLED", True),
        help="Enable or disable the Wyoming protocol server.",
    )
    parser.add_argument(
        "--wyoming-uri",
        default=environment_default("FATTERQWEN_WYOMING_URI", "tcp://0.0.0.0:10300"),
        help="Wyoming server URI such as tcp://0.0.0.0:10300 or unix:///tmp/fatterqwen.sock.",
    )
    parser.add_argument(
        "--wyoming-audio-chunk-samples",
        type=int,
        default=int(environment_default("FATTERQWEN_WYOMING_AUDIO_CHUNK_SAMPLES", "4096")),
        help="How many mono PCM samples to pack into each Wyoming audio-chunk event.",
    )
    parser.add_argument(
        "--log-level",
        default=environment_default("FATTERQWEN_LOG_LEVEL", "INFO"),
        help="Python logging level.",
    )
    return parser



def parse_server_config(argv: Optional[Sequence[str]] = None) -> ServerConfig:
    """Parse CLI arguments and normalize them into the immutable runtime config.

    Usage:
        Call this once at startup from the CLI entry point. The function validates
        a few basic invariants early so server startup fails fast on obviously
        invalid configuration.

    Parameters:
        argv: Optional explicit CLI argument list. When omitted, argparse reads
        the process command line.

    Returns:
        A `ServerConfig` object ready to pass into the runtime layer.

    Raises:
        SystemExit: Raised by argparse when validation fails.
    """
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.port <= 0:
        parser.error("--port must be a positive integer.")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be a positive integer.")
    if args.max_text_length <= 0:
        parser.error("--max-text-length must be a positive integer.")
    if args.reference_prompt_target_rms < 0.0:
        parser.error("--reference-prompt-target-rms must be zero or positive.")
    if args.longform_chunk_threshold_units <= 0:
        parser.error("--longform-chunk-threshold-units must be a positive integer.")
    if args.longform_target_units <= 0:
        parser.error("--longform-target-units must be a positive integer.")
    if args.longform_min_units <= 0:
        parser.error("--longform-min-units must be a positive integer.")
    if args.longform_crossfade_milliseconds < 0:
        parser.error("--longform-crossfade-milliseconds must be zero or positive.")
    if args.longform_gap_milliseconds < 0:
        parser.error("--longform-gap-milliseconds must be zero or positive.")
    if args.wyoming_audio_chunk_samples <= 0:
        parser.error("--wyoming-audio-chunk-samples must be a positive integer.")

    model_cache_dir = None
    if args.model_cache_dir:
        model_cache_dir = Path(args.model_cache_dir).expanduser().resolve()

    prefetch_manifest_path = None
    if args.prefetch_manifest:
        prefetch_manifest_path = Path(args.prefetch_manifest).expanduser().resolve()

    wyoming_uri: Optional[str] = args.wyoming_uri
    if args.wyoming_enabled and not wyoming_uri:
        parser.error("--wyoming-uri must be set when Wyoming support is enabled.")
    if not args.wyoming_enabled:
        wyoming_uri = None

    return ServerConfig(
        voices_dir=Path(args.voices_dir).expanduser().resolve(),
        host=args.host,
        port=args.port,
        model=args.model,
        device=args.device,
        dtype=args.dtype,
        default_language=args.default_language,
        chunk_size=args.chunk_size,
        append_silence=args.append_silence,
        non_streaming_mode=args.non_streaming_mode,
        max_text_length=args.max_text_length,
        model_cache_dir=model_cache_dir,
        prefetch_manifest_path=prefetch_manifest_path,
        warmup=args.warmup,
        warmup_text=args.warmup_text,
        wyoming_enabled=args.wyoming_enabled,
        wyoming_uri=wyoming_uri,
        wyoming_audio_chunk_samples=args.wyoming_audio_chunk_samples,
        log_level=args.log_level.upper(),
        preprocess_reference_audio=args.preprocess_reference_audio,
        normalize_reference_transcript=args.normalize_reference_transcript,
        reference_prompt_target_rms=args.reference_prompt_target_rms,
        postprocess_output_audio=args.postprocess_output_audio,
        longform_chunking_enabled=args.longform_chunking,
        longform_chunk_threshold_units=args.longform_chunk_threshold_units,
        longform_target_units=args.longform_target_units,
        longform_min_units=args.longform_min_units,
        longform_crossfade_milliseconds=args.longform_crossfade_milliseconds,
        longform_gap_milliseconds=args.longform_gap_milliseconds,
    )
