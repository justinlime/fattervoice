"""CLI and environment-based configuration for the fattervoice server."""

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
    max_text_length: int
    model_cache_dir: Optional[Path]
    prefetch_manifest_path: Optional[Path]
    warmup: bool
    warmup_text: str
    wyoming_enabled: bool
    wyoming_uri: Optional[str]
    wyoming_audio_chunk_samples: int
    log_level: str
    num_step: int = 16
    guidance_scale: float = 2.0
    denoise: bool = True
    t_shift: float = 0.1
    position_temperature: float = 5.0
    class_temperature: float = 0.0
    layer_penalty_factor: float = 5.0
    preprocess_voice_clone_prompt: bool = True
    postprocess_output_audio: bool = True
    audio_chunk_duration: float = 15.0
    audio_chunk_threshold: float = 30.0



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
    """Construct the command-line parser for the production OmniVoice server.

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
        description="Production-oriented OpenAI-compatible and Wyoming TTS server for OmniVoice.",
    )
    parser.add_argument(
        "--voices-dir",
        default=environment_default("FATTERVOICE_VOICES_DIR", "voices"),
        help="Directory containing <voice>.<audio> and <voice>.txt pairs.",
    )
    parser.add_argument(
        "--host",
        default=environment_default("FATTERVOICE_HOST", "0.0.0.0"),
        help="HTTP bind host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(environment_default("FATTERVOICE_PORT", "8000")),
        help="HTTP bind port.",
    )
    parser.add_argument(
        "--model",
        default=environment_default("FATTERVOICE_MODEL", "omnivoice"),
        help="Model alias (`omnivoice`) or a direct model ID/path.",
    )
    parser.add_argument(
        "--device",
        default=environment_default("FATTERVOICE_DEVICE", "cuda:0"),
        help="Device map passed through to OmniVoice, such as cuda:0, cpu, or mps.",
    )
    parser.add_argument(
        "--dtype",
        default=environment_default("FATTERVOICE_DTYPE", "float16"),
        help="Torch dtype name such as float16, bfloat16, or float32.",
    )
    parser.add_argument(
        "--default-language",
        default=environment_default("FATTERVOICE_DEFAULT_LANGUAGE", "auto"),
        help="Fallback language ID or name passed to OmniVoice when the client does not specify one.",
    )
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=int(environment_default("FATTERVOICE_MAX_TEXT_LENGTH", "4000")),
        help="Maximum request text length accepted by the server.",
    )
    parser.add_argument(
        "--num-step",
        type=int,
        default=int(environment_default("FATTERVOICE_NUM_STEP", "16")),
        help="OmniVoice diffusion decoding steps. Lower is faster; 16 is the default speed-focused setting.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=float(environment_default("FATTERVOICE_GUIDANCE_SCALE", "2.0")),
        help="OmniVoice classifier-free guidance scale.",
    )
    parser.add_argument(
        "--denoise",
        action=argparse.BooleanOptionalAction,
        default=environment_flag("FATTERVOICE_DENOISE", True),
        help="Enable OmniVoice denoise prompting for cleaner output.",
    )
    parser.add_argument(
        "--t-shift",
        type=float,
        default=float(environment_default("FATTERVOICE_T_SHIFT", "0.1")),
        help="OmniVoice diffusion time-shift parameter.",
    )
    parser.add_argument(
        "--position-temperature",
        type=float,
        default=float(environment_default("FATTERVOICE_POSITION_TEMPERATURE", "5.0")),
        help="OmniVoice position-sampling temperature.",
    )
    parser.add_argument(
        "--class-temperature",
        type=float,
        default=float(environment_default("FATTERVOICE_CLASS_TEMPERATURE", "0.0")),
        help="OmniVoice class-sampling temperature. Zero keeps decoding deterministic.",
    )
    parser.add_argument(
        "--layer-penalty-factor",
        type=float,
        default=float(environment_default("FATTERVOICE_LAYER_PENALTY_FACTOR", "5.0")),
        help="Penalty factor that biases OmniVoice toward lower codebook layers first.",
    )
    parser.add_argument(
        "--preprocess-voice-clone-prompt",
        action=argparse.BooleanOptionalAction,
        default=environment_flag("FATTERVOICE_PREPROCESS_VOICE_CLONE_PROMPT", True),
        help="Preprocess reference audio and transcript once before caching each OmniVoice voice-clone prompt.",
    )
    parser.add_argument(
        "--postprocess-output-audio",
        action=argparse.BooleanOptionalAction,
        default=environment_flag("FATTERVOICE_POSTPROCESS_OUTPUT_AUDIO", True),
        help="Allow OmniVoice to remove excess silence and pad/fade output audio.",
    )
    parser.add_argument(
        "--audio-chunk-duration",
        type=float,
        default=float(environment_default("FATTERVOICE_AUDIO_CHUNK_DURATION", "15.0")),
        help="Target per-chunk duration in seconds for OmniVoice long-form generation.",
    )
    parser.add_argument(
        "--audio-chunk-threshold",
        type=float,
        default=float(environment_default("FATTERVOICE_AUDIO_CHUNK_THRESHOLD", "30.0")),
        help="Estimated duration threshold in seconds above which OmniVoice chunked long-form generation is activated.",
    )
    parser.add_argument(
        "--model-cache-dir",
        default=(
            environment_default("FATTERVOICE_MODEL_CACHE_DIR", "")
            or environment_default("HF_HUB_CACHE", "")
            or environment_default("HUGGINGFACE_HUB_CACHE", "")
            or environment_default("TRANSFORMERS_CACHE", "")
        ),
        help="Optional Hugging Face hub cache directory for model artifacts.",
    )
    parser.add_argument(
        "--prefetch-manifest",
        default=environment_default("FATTERVOICE_PREFETCH_MANIFEST", ""),
        help="Optional JSON manifest that maps prefetched model IDs to exact local snapshot paths.",
    )
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=environment_flag("FATTERVOICE_WARMUP", False),
        help="Run a short warmup generation during startup before traffic arrives.",
    )
    parser.add_argument(
        "--warmup-text",
        default=environment_default("FATTERVOICE_WARMUP_TEXT", "Hello from fattervoice."),
        help="Text used when warmup is enabled.",
    )
    parser.add_argument(
        "--wyoming-enabled",
        action=argparse.BooleanOptionalAction,
        default=environment_flag("FATTERVOICE_WYOMING_ENABLED", True),
        help="Enable or disable the Wyoming protocol server.",
    )
    parser.add_argument(
        "--wyoming-uri",
        default=environment_default("FATTERVOICE_WYOMING_URI", "tcp://0.0.0.0:10300"),
        help="Wyoming server URI such as tcp://0.0.0.0:10300 or unix:///tmp/fattervoice.sock.",
    )
    parser.add_argument(
        "--wyoming-audio-chunk-samples",
        type=int,
        default=int(environment_default("FATTERVOICE_WYOMING_AUDIO_CHUNK_SAMPLES", "4096")),
        help="How many mono PCM samples to pack into each Wyoming audio-chunk event.",
    )
    parser.add_argument(
        "--log-level",
        default=environment_default("FATTERVOICE_LOG_LEVEL", "INFO"),
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
    if args.max_text_length <= 0:
        parser.error("--max-text-length must be a positive integer.")
    if args.num_step <= 0:
        parser.error("--num-step must be a positive integer.")
    if args.guidance_scale < 0.0:
        parser.error("--guidance-scale must be zero or positive.")
    if args.position_temperature < 0.0:
        parser.error("--position-temperature must be zero or positive.")
    if args.class_temperature < 0.0:
        parser.error("--class-temperature must be zero or positive.")
    if args.layer_penalty_factor < 0.0:
        parser.error("--layer-penalty-factor must be zero or positive.")
    if args.audio_chunk_duration <= 0.0:
        parser.error("--audio-chunk-duration must be greater than zero.")
    if args.audio_chunk_threshold <= 0.0:
        parser.error("--audio-chunk-threshold must be greater than zero.")
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
        max_text_length=args.max_text_length,
        model_cache_dir=model_cache_dir,
        prefetch_manifest_path=prefetch_manifest_path,
        warmup=args.warmup,
        warmup_text=args.warmup_text,
        wyoming_enabled=args.wyoming_enabled,
        wyoming_uri=wyoming_uri,
        wyoming_audio_chunk_samples=args.wyoming_audio_chunk_samples,
        log_level=args.log_level.upper(),
        num_step=args.num_step,
        guidance_scale=args.guidance_scale,
        denoise=args.denoise,
        t_shift=args.t_shift,
        position_temperature=args.position_temperature,
        class_temperature=args.class_temperature,
        layer_penalty_factor=args.layer_penalty_factor,
        preprocess_voice_clone_prompt=args.preprocess_voice_clone_prompt,
        postprocess_output_audio=args.postprocess_output_audio,
        audio_chunk_duration=args.audio_chunk_duration,
        audio_chunk_threshold=args.audio_chunk_threshold,
    )
