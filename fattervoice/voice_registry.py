"""Voice directory scanning and validation for basename-paired reference voices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional
import wave

_AUDIO_SUFFIXES = {".aif", ".aiff", ".au", ".flac", ".ogg", ".wav"}
_REFERENCE_TRANSCRIPT_SUFFIX = ".ref.txt"
_INSTRUCT_TEXT_SUFFIX = ".instruct.txt"


class VoiceRegistryError(ValueError):
    """Raised when the configured voices directory does not satisfy the project contract."""


@dataclass(frozen=True)
class VoiceEntry:
    """Immutable metadata for a single validated voice entry."""

    voice_id: str
    audio_path: Path
    transcript_path: Path
    transcript: str
    instruct_path: Path | None = None
    instruct: str | None = None



def parse_voice_text_metadata_path(metadata_path: Path) -> tuple[str, str] | None:
    """Classify a voice text file as required transcript or optional instruct metadata.

    Usage:
        Voice-registry scanning calls this helper for every discovered text file
        so compound suffixes like `.ref.txt` and `.instruct.txt` are parsed
        consistently without relying on `Path.stem`, which would otherwise drop
        only the final `.txt` suffix.

    Parameters:
        metadata_path: The file path that should be inspected.

    Returns:
        A `(voice_id, metadata_kind)` tuple when the file matches a supported
        voice text-metadata suffix, otherwise `None`.
    """
    filename = metadata_path.name
    for suffix, metadata_kind in (
        (_REFERENCE_TRANSCRIPT_SUFFIX, "transcript"),
        (_INSTRUCT_TEXT_SUFFIX, "instruct"),
    ):
        if filename.endswith(suffix):
            voice_id = filename[: -len(suffix)]
            return (voice_id, metadata_kind) if voice_id else None
    return None



def read_optional_instruct_text(instruct_path: Path | None) -> str | None:
    """Load optional per-voice OmniVoice instruct text from disk when present.

    Usage:
        Voice-registry scanning uses this helper after file pairing so optional
        `.instruct.txt` files can be normalized once at startup and then reused
        for every request that selects the associated voice.

    Parameters:
        instruct_path: The optional instruct-text file path paired with a voice.

    Returns:
        The stripped instruct string when a non-empty file exists, otherwise
        `None` when no instruct file is configured or the file is blank.
    """
    if instruct_path is None:
        return None

    normalized_instruct = instruct_path.read_text(encoding="utf-8").strip()
    return normalized_instruct or None


class VoiceRegistry:
    """Validated mapping from public voice IDs to reference voice assets."""

    def __init__(self, voices_dir: Path, entries: Iterable[VoiceEntry]) -> None:
        """Create an in-memory registry from already validated voice entries.

        Usage:
            The public constructor is mainly used internally by `scan`. Callers
            typically use `VoiceRegistry.scan(...)` so validation happens before
            an instance is returned.

        Parameters:
            voices_dir: The absolute directory that contains the discovered voice files.
            entries: Validated voice entries that should populate the registry.

        Returns:
            None. A new `VoiceRegistry` instance is initialized in place.
        """
        ordered_entries = sorted(entries, key=lambda entry: entry.voice_id)
        if not ordered_entries:
            raise VoiceRegistryError("Voice registry cannot be empty.")

        self.voices_dir = voices_dir
        self._entries: Dict[str, VoiceEntry] = {
            entry.voice_id: entry for entry in ordered_entries
        }
        self.default_voice_id = ordered_entries[0].voice_id

    @classmethod
    def scan(cls, voices_dir: Path) -> "VoiceRegistry":
        """Scan a voices directory and build a fully validated voice registry.

        Usage:
            Call this during startup so the server can fail fast on missing
            `.ref.txt` transcripts, duplicate basenames, stray metadata files,
            unsupported layouts, or unreadable reference audio files.

        Parameters:
            voices_dir: The directory containing reference audio plus paired text files.

        Returns:
            A ready-to-use `VoiceRegistry` containing all discovered voices.

        Raises:
            VoiceRegistryError: If the directory is missing or contains invalid pairs.
        """
        if not voices_dir.exists():
            raise VoiceRegistryError(f"Voices directory does not exist: {voices_dir}")
        if not voices_dir.is_dir():
            raise VoiceRegistryError(f"Voices path is not a directory: {voices_dir}")

        audio_files: Dict[str, Path] = {}
        transcript_files: Dict[str, Path] = {}
        instruct_files: Dict[str, Path] = {}

        for child_path in sorted(voices_dir.iterdir()):
            if not child_path.is_file():
                continue

            metadata_match = parse_voice_text_metadata_path(child_path)
            if metadata_match is not None:
                voice_id, metadata_kind = metadata_match
                if metadata_kind == "transcript":
                    if voice_id in transcript_files:
                        raise VoiceRegistryError(
                            "Duplicate reference transcript files found for voice "
                            f"{voice_id!r}: {transcript_files[voice_id].name} and {child_path.name}."
                        )
                    transcript_files[voice_id] = child_path
                else:
                    if voice_id in instruct_files:
                        raise VoiceRegistryError(
                            "Duplicate instruct text files found for voice "
                            f"{voice_id!r}: {instruct_files[voice_id].name} and {child_path.name}."
                        )
                    instruct_files[voice_id] = child_path
                continue

            suffix = child_path.suffix.lower()
            voice_id = child_path.stem
            if suffix not in _AUDIO_SUFFIXES:
                continue

            if voice_id in audio_files:
                raise VoiceRegistryError(
                    "Duplicate audio files found for voice "
                    f"{voice_id!r}: {audio_files[voice_id].name} and {child_path.name}."
                )

            audio_files[voice_id] = child_path

        if not audio_files:
            raise VoiceRegistryError(
                f"No supported voice audio files were found in {voices_dir}."
            )

        missing_transcripts = sorted(set(audio_files) - set(transcript_files))
        if missing_transcripts:
            raise VoiceRegistryError(
                "Missing reference transcript files for voices: "
                + ", ".join(
                    f"{voice_id}{_REFERENCE_TRANSCRIPT_SUFFIX}"
                    for voice_id in missing_transcripts
                )
            )

        stray_transcripts = sorted(set(transcript_files) - set(audio_files))
        if stray_transcripts:
            raise VoiceRegistryError(
                "Reference transcript files without matching audio were found for voices: "
                + ", ".join(
                    f"{voice_id}{_REFERENCE_TRANSCRIPT_SUFFIX}"
                    for voice_id in stray_transcripts
                )
            )

        stray_instructs = sorted(set(instruct_files) - set(audio_files))
        if stray_instructs:
            raise VoiceRegistryError(
                "Instruct text files without matching audio were found for voices: "
                + ", ".join(
                    f"{voice_id}{_INSTRUCT_TEXT_SUFFIX}"
                    for voice_id in stray_instructs
                )
            )

        entries = []
        for voice_id, audio_path in sorted(audio_files.items()):
            transcript_path = transcript_files[voice_id]
            transcript = transcript_path.read_text(encoding="utf-8").strip()
            if not transcript:
                raise VoiceRegistryError(
                    f"Reference transcript file is empty for voice {voice_id!r}: {transcript_path}"
                )

            instruct_path = instruct_files.get(voice_id)
            instruct = read_optional_instruct_text(instruct_path)

            validate_audio_file(audio_path)
            entries.append(
                VoiceEntry(
                    voice_id=voice_id,
                    audio_path=audio_path,
                    transcript_path=transcript_path,
                    transcript=transcript,
                    instruct_path=instruct_path,
                    instruct=instruct,
                )
            )

        return cls(voices_dir=voices_dir, entries=entries)

    def get(self, voice_id: Optional[str]) -> VoiceEntry:
        """Resolve a public voice ID to a validated voice entry.

        Usage:
            API and Wyoming adapters call this to map client-supplied voice names
            onto the canonical registry entries discovered at startup.

        Parameters:
            voice_id: The requested voice identifier. `None` selects the default voice.

        Returns:
            The matching `VoiceEntry`.

        Raises:
            VoiceRegistryError: If the requested voice does not exist.
        """
        requested_voice_id = voice_id or self.default_voice_id
        try:
            return self._entries[requested_voice_id]
        except KeyError as exc:
            raise VoiceRegistryError(
                "Unknown voice "
                f"{requested_voice_id!r}. Available voices: {', '.join(self.list_voice_ids())}"
            ) from exc

    def list_voice_ids(self) -> list[str]:
        """Return the stable, sorted list of public voice identifiers.

        Usage:
            This is used by both HTTP discovery endpoints and the Wyoming info
            advertisement so every protocol layer exposes the same canonical names.

        Parameters:
            None.

        Returns:
            A sorted list of voice IDs.
        """
        return sorted(self._entries)

    def values(self) -> list[VoiceEntry]:
        """Return every validated voice entry in stable voice-ID order.

        Usage:
            Callers that need richer metadata than `list_voice_ids` can use this
            method to iterate over the full registry contents.

        Parameters:
            None.

        Returns:
            A list of `VoiceEntry` objects sorted by voice ID.
        """
        return [self._entries[voice_id] for voice_id in self.list_voice_ids()]



def validate_audio_file(audio_path: Path) -> None:
    """Validate that a discovered reference audio file is readable by runtime dependencies.

    Usage:
        Startup validation calls this helper so broken or unsupported voice files
        fail immediately instead of during the first user request.

    Parameters:
        audio_path: The audio file path that should be inspected.

    Returns:
        None. The function succeeds silently when the file is readable.

    Raises:
        VoiceRegistryError: If the file is empty or cannot be parsed as audio.
    """
    suffix = audio_path.suffix.lower()

    if suffix == ".wav":
        try:
            with wave.open(str(audio_path), "rb") as wav_file:
                if wav_file.getnframes() <= 0:
                    raise VoiceRegistryError(f"WAV file has no audio frames: {audio_path}")
                return
        except wave.Error as exc:
            raise VoiceRegistryError(f"Unreadable WAV file {audio_path}: {exc}") from exc

    try:
        import soundfile as sf
    except ImportError as exc:
        raise VoiceRegistryError(
            "soundfile is required to validate non-WAV reference audio files. "
            f"Unable to validate {audio_path}."
        ) from exc

    try:
        info = sf.info(str(audio_path))
    except Exception as exc:  # pragma: no cover - exact exception types vary by backend.
        raise VoiceRegistryError(f"Unreadable audio file {audio_path}: {exc}") from exc

    if info.frames <= 0:
        raise VoiceRegistryError(f"Audio file has no samples: {audio_path}")
