"""CLI and environment-based configuration for the fattervoice server."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


def build_tcp_uri(host: str, port: int) -> str:
    """Build a TCP URI string from a host and port, including IPv6-safe formatting.

    Usage:
        The runtime exposes split host/port configuration for both OpenAI-compatible
        HTTP and Wyoming networking, but the Wyoming server still expects one final
        `tcp://...` URI string. This helper keeps that assembly logic centralized
        and ensures IPv6 literals are wrapped in brackets before the port suffix is
        appended.

    Parameters:
        host: The configured bind host or address literal.
        port: The configured TCP port.

    Returns:
        A `tcp://host:port` URI string, with IPv6 literals normalized to the
        bracketed URI form.
    """
    normalized_host = host.strip()
    if ":" in normalized_host and not normalized_host.startswith("["):
        normalized_host = f"[{normalized_host}]"
    return f"tcp://{normalized_host}:{port}"


@dataclass(frozen=True)
class ServerConfig:
    """Immutable runtime configuration shared by the HTTP and Wyoming servers."""

    voices_dir: Path
    openapi_host: str
    openapi_port: int
    device: str
    dtype: str
    default_language: str
    wyoming_host: str
    wyoming_port: int
    log_level: str
    num_step: int = 32
    guidance_scale: float = 2.0
    denoise: bool = True
    t_shift: float = 0.1
    position_temperature: float = 5.0
    class_temperature: float = 0.0
    layer_penalty_factor: float = 5.0
    preprocess_voice_clone_prompt: bool = True
    postprocess_output_audio: bool = True
    max_sentence_length: int = 400
    break_point_lookback: int = 100
    preload_voice: str | None = None

    @property
    def wyoming_uri(self) -> str:
        """Build the Wyoming TCP URI from the split host and port settings.

        Usage:
            The external configuration surface now exposes Wyoming host and port
            as separate values, while the server runtime still needs one final
            URI string to pass into `wyoming.AsyncServer.from_uri(...)`.

        Parameters:
            None.

        Returns:
            A `tcp://host:port` URI string suitable for the Wyoming server,
            including bracketed formatting for IPv6 literals.
        """
        return build_tcp_uri(self.wyoming_host, self.wyoming_port)



def format_config_value(value: object) -> str:
    """Convert a resolved configuration value into a stable startup-log string.

    Usage:
        Startup summary rendering calls this helper for every effective config
        value so booleans, paths, and scalar settings all display consistently.

    Parameters:
        value: The already-resolved configuration value that should be rendered.

    Returns:
        A user-friendly string representation suitable for multi-line startup
        logging output.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)



def format_server_config_summary(config: ServerConfig) -> str:
    """Render the full effective server configuration as a boxed multi-line string.

    Usage:
        The CLI logs this summary once during startup so operators can quickly
        verify the exact runtime settings that won after CLI parsing and
        environment fallback resolution.

    Parameters:
        config: The fully resolved immutable runtime configuration.

    Returns:
        A boxed multi-line string that groups every public config value into
        readable sections for startup logs.
    """
    section_rows = (
        (
            "Network",
            (
                ("OpenAI host", config.openapi_host),
                ("OpenAI port", config.openapi_port),
                ("Wyoming host", config.wyoming_host),
                ("Wyoming port", config.wyoming_port),
            ),
        ),
        (
            "Runtime",
            (
                ("Voices directory", config.voices_dir),
                ("Preload voice", config.preload_voice or "(none)"),
                ("Device", config.device),
                ("Dtype", config.dtype),
                ("Log level", config.log_level),
                ("Default language", config.default_language),
            ),
        ),
        (
            "Generation",
            (
                ("num_step", config.num_step),
                ("guidance_scale", config.guidance_scale),
                ("denoise", config.denoise),
                ("t_shift", config.t_shift),
                ("position_temperature", config.position_temperature),
                ("class_temperature", config.class_temperature),
                ("layer_penalty_factor", config.layer_penalty_factor),
                ("preprocess_voice_clone_prompt", config.preprocess_voice_clone_prompt),
                ("postprocess_output_audio", config.postprocess_output_audio),
                ("max_sentence_length", config.max_sentence_length),
            ),
        ),
    )
    label_width = max(len(label) for _, rows in section_rows for label, _ in rows)
    content_lines: list[str] = []

    for section_index, (section_name, rows) in enumerate(section_rows):
        if section_index:
            content_lines.append("")
        content_lines.append(section_name)
        for label, value in rows:
            content_lines.append(
                f"  {label.ljust(label_width)} : {format_config_value(value)}"
            )

    title = "fattervoice startup configuration"
    inner_width = max(len(title), *(len(line) for line in content_lines))
    top_border = f"╭{'─' * (inner_width + 2)}╮"
    title_line = f"│ {title.ljust(inner_width)} │"
    separator_line = f"├{'─' * (inner_width + 2)}┤"
    body_lines = [f"│ {line.ljust(inner_width)} │" for line in content_lines]
    bottom_border = f"╰{'─' * (inner_width + 2)}╯"
    return "\n".join([top_border, title_line, separator_line, *body_lines, bottom_border])



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
        "--openapi-host",
        default=environment_default("FATTERVOICE_OPENAPI_HOST", "0.0.0.0"),
        help="OpenAI-compatible HTTP bind host.",
    )
    parser.add_argument(
        "--openapi-port",
        type=int,
        default=int(environment_default("FATTERVOICE_OPENAPI_PORT", "8000")),
        help="OpenAI-compatible HTTP bind port.",
    )
    parser.add_argument(
        "--device",
        default=environment_default("FATTERVOICE_DEVICE", "cuda:0"),
        help="Device map passed through to OmniVoice, such as cuda:0, cpu, or mps.",
    )
    parser.add_argument(
        "--dtype",
        default=environment_default("FATTERVOICE_DTYPE", "bfloat16"),
        help="Torch dtype name such as float16, bfloat16, or float32.",
    )
    parser.add_argument(
        "--default-language",
        default=environment_default("FATTERVOICE_DEFAULT_LANGUAGE", "auto"),
        help="Fallback language ID or name passed to OmniVoice when the client does not specify one.",
    )
    parser.add_argument(
        "--num-step",
        type=int,
        default=int(environment_default("FATTERVOICE_NUM_STEP", "32")),
        help="OmniVoice diffusion decoding steps. Lower is faster; 32 is the default balanced setting.",
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
        "--max-sentence-length",
        type=int,
        default=int(environment_default("FATTERVOICE_MAX_SENTENCE_LENGTH", "400")),
        help="Maximum character length of a single synthesis segment before it is split further. Sentence boundaries are respected first; only segments exceeding this cap are broken on spaces. Default 400 (~25s of speech).",
    )
    parser.add_argument(
        "--wyoming-host",
        default=environment_default("FATTERVOICE_WYOMING_HOST", "0.0.0.0"),
        help="Wyoming TCP bind host.",
    )
    parser.add_argument(
        "--wyoming-port",
        type=int,
        default=int(environment_default("FATTERVOICE_WYOMING_PORT", "10300")),
        help="Wyoming TCP bind port.",
    )
    parser.add_argument(
        "--preload-voice",
        default=environment_default("FATTERVOICE_PRELOAD_VOICE", ""),
        help="Voice ID to pre-load its clone prompt at startup (e.g. hank).",
    )
    parser.add_argument(
        "--log-level",
        default=environment_default("FATTERVOICE_LOG_LEVEL", "INFO"),
        help="Python logging level.",
    )
    return parser



def parse_server_config(argv: Sequence[str] | None = None) -> ServerConfig:
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

    if args.openapi_port <= 0:
        parser.error("--openapi-port must be a positive integer.")
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
    if args.max_sentence_length <= 0:
        parser.error("--max-sentence-length must be greater than zero.")
    if args.wyoming_port <= 0:
        parser.error("--wyoming-port must be a positive integer.")

    preload_voice_raw = args.preload_voice.strip() if args.preload_voice else ""
    preload_voice = preload_voice_raw if preload_voice_raw else None

    return ServerConfig(
        voices_dir=Path(args.voices_dir).expanduser().resolve(),
        openapi_host=args.openapi_host,
        openapi_port=args.openapi_port,
        device=args.device,
        dtype=args.dtype,
        default_language=args.default_language,
        wyoming_host=args.wyoming_host,
        wyoming_port=args.wyoming_port,
        log_level=args.log_level.upper(),
        preload_voice=preload_voice,
        num_step=args.num_step,
        guidance_scale=args.guidance_scale,
        denoise=args.denoise,
        t_shift=args.t_shift,
        position_temperature=args.position_temperature,
        class_temperature=args.class_temperature,
        layer_penalty_factor=args.layer_penalty_factor,
        preprocess_voice_clone_prompt=args.preprocess_voice_clone_prompt,
        postprocess_output_audio=args.postprocess_output_audio,
        max_sentence_length=args.max_sentence_length,
    )
