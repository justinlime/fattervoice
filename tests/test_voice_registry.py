"""Unit tests for the validated voices directory registry."""

from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

from fattervoice.voice_registry import VoiceRegistry, VoiceRegistryError



def write_test_wav(path: Path) -> None:
    """Create a tiny mono WAV file for registry validation tests.

    Usage:
        The registry tests use this helper to create a real readable audio file
        without requiring any third-party audio libraries.

    Parameters:
        path: The WAV file path that should be created.

    Returns:
        None. The function writes a short valid WAV file to disk.
    """
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(b"\x00\x00" * 240)


class VoiceRegistryTests(unittest.TestCase):
    """Verify that the registry enforces the basename-paired voice directory contract."""

    def test_scan_valid_voice_directory_without_instruct_file(self) -> None:
        """Ensure a required audio/ref-transcript pair still works without instruct text.

        Usage:
            This test protects the backward-compatible happy path where a voice
            provides the required audio plus `.ref.txt` transcript but omits the
            optional `.instruct.txt` companion.

        Parameters:
            None.

        Returns:
            None. The test asserts that the registry exposes the expected voice
            ID and leaves the optional instruct fields unset.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            voices_dir = Path(temp_dir)
            write_test_wav(voices_dir / "hank.wav")
            (voices_dir / "hank.ref.txt").write_text("Reference transcript.", encoding="utf-8")

            registry = VoiceRegistry.scan(voices_dir)
            voice_entry = registry.get("hank")

            self.assertEqual(registry.default_voice_id, "hank")
            self.assertEqual(voice_entry.transcript, "Reference transcript.")
            self.assertIsNone(voice_entry.instruct_path)
            self.assertIsNone(voice_entry.instruct)

    def test_scan_preserves_optional_instruct_text_when_present(self) -> None:
        """Ensure optional instruct metadata is loaded and attached to the voice.

        Usage:
            OmniVoice voice design uses an optional `instruct` string, so this
            test verifies that `<voice>.instruct.txt` is discovered at startup
            and stored on the resulting `VoiceEntry` for later synthesis calls.

        Parameters:
            None.

        Returns:
            None. The test asserts that the registry exposes the instruct path
            and stripped instruct text for the selected voice.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            voices_dir = Path(temp_dir)
            write_test_wav(voices_dir / "hank.wav")
            (voices_dir / "hank.ref.txt").write_text("Reference transcript.", encoding="utf-8")
            (voices_dir / "hank.instruct.txt").write_text(
                " female, british accent, whisper \n",
                encoding="utf-8",
            )

            registry = VoiceRegistry.scan(voices_dir)
            voice_entry = registry.get("hank")

            self.assertEqual(voice_entry.transcript_path.name, "hank.ref.txt")
            self.assertEqual(voice_entry.instruct_path.name, "hank.instruct.txt")
            self.assertEqual(voice_entry.instruct, "female, british accent, whisper")

    def test_scan_rejects_missing_reference_transcript(self) -> None:
        """Ensure startup validation fails when an audio file lacks a matching `.ref.txt`.

        Usage:
            This test protects the project's voice-cloning quality contract that
            every voice must provide a reference transcript file using the new
            `.ref.txt` naming convention.

        Parameters:
            None.

        Returns:
            None. The test asserts that a `VoiceRegistryError` is raised.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            voices_dir = Path(temp_dir)
            write_test_wav(voices_dir / "hank.wav")

            with self.assertRaises(VoiceRegistryError):
                VoiceRegistry.scan(voices_dir)

    def test_scan_rejects_duplicate_audio_basenames(self) -> None:
        """Ensure startup validation fails when multiple audio files share one basename.

        Usage:
            This test protects the project contract that duplicate voice IDs must
            be treated as configuration errors rather than silently picking one.

        Parameters:
            None.

        Returns:
            None. The test asserts that a `VoiceRegistryError` is raised.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            voices_dir = Path(temp_dir)
            write_test_wav(voices_dir / "hank.wav")
            write_test_wav(voices_dir / "hank.aiff")
            (voices_dir / "hank.ref.txt").write_text("Reference transcript.", encoding="utf-8")

            with self.assertRaises(VoiceRegistryError):
                VoiceRegistry.scan(voices_dir)


if __name__ == "__main__":
    unittest.main()
