"""Unit tests for the validated voices directory registry."""

from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

from fatterqwen.voice_registry import VoiceRegistry, VoiceRegistryError



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

    def test_scan_valid_voice_directory(self) -> None:
        """Ensure a complete audio/transcript pair becomes a discoverable voice entry.

        Usage:
            This test protects the happy path that server startup depends on when
            it scans the configured `voices/` directory.

        Parameters:
            None.

        Returns:
            None. The test asserts that the registry exposes the expected voice ID.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            voices_dir = Path(temp_dir)
            write_test_wav(voices_dir / "hank.wav")
            (voices_dir / "hank.txt").write_text("Reference transcript.", encoding="utf-8")

            registry = VoiceRegistry.scan(voices_dir)

            self.assertEqual(registry.default_voice_id, "hank")
            self.assertEqual(registry.get("hank").transcript, "Reference transcript.")

    def test_scan_rejects_missing_transcript(self) -> None:
        """Ensure startup validation fails when an audio file lacks a matching transcript.

        Usage:
            This test protects the project's voice-cloning quality contract that
            every voice must provide a transcript.

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
            (voices_dir / "hank.txt").write_text("Reference transcript.", encoding="utf-8")

            with self.assertRaises(VoiceRegistryError):
                VoiceRegistry.scan(voices_dir)


if __name__ == "__main__":
    unittest.main()
