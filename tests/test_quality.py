"""Unit tests for standalone quality helpers used by the synthesis service."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from fatterqwen.quality import (
    clean_generated_audio,
    compact_silences,
    compute_audio_rms,
    ensure_terminal_punctuation,
    merge_audio_segments_with_crossfade,
    prepare_voice_conditioning,
    split_audio_tail_for_crossfade,
    split_text_for_longform_synthesis,
)


class QualityHelperTests(unittest.TestCase):
    """Verify that standalone quality helpers behave predictably on synthetic data."""

    def test_split_text_for_longform_synthesis_preserves_ordered_sentences(self) -> None:
        """Ensure long-form chunking breaks large text at natural sentence boundaries.

        Usage:
            This test guards the wrapper's home-grown chunk planner so long-form
            requests are split into ordered sub-prompts instead of arbitrary raw
            slices.

        Parameters:
            None.

        Returns:
            None. The test asserts that multiple chunks are produced and that the
            original sentence order is preserved.
        """
        text = "Alpha sentence. Beta sentence. Gamma sentence. Delta sentence."

        text_chunks = split_text_for_longform_synthesis(
            text,
            threshold_units=12,
            target_units=10,
            min_units=4,
        )

        self.assertGreater(len(text_chunks), 1)
        self.assertEqual(
            "".join(chunk.replace(" ", "") for chunk in text_chunks),
            text.replace(" ", ""),
        )

    def test_ensure_terminal_punctuation_inserts_period_before_trailing_quotes(self) -> None:
        """Ensure punctuation normalization respects trailing closing punctuation marks.

        Usage:
            Reference transcripts can end with quotes or brackets, so this test
            verifies that a missing full stop is inserted before those closers
            instead of being appended after them.

        Parameters:
            None.

        Returns:
            None. The test asserts the normalized punctuation placement.
        """
        self.assertEqual(
            ensure_terminal_punctuation('He said "hello"'),
            'He said "hello."',
        )
        self.assertEqual(
            ensure_terminal_punctuation('He said "hello."'),
            'He said "hello."',
        )

    def test_prepare_voice_conditioning_compacts_silence_and_normalizes_transcript(self) -> None:
        """Ensure reference preparation removes dead air and boosts very quiet prompts.

        Usage:
            The voice-cloning service depends on cached prepared prompt audio. This
            test verifies that preparation writes a cleaned file, keeps track of
            RMS, and normalizes missing transcript punctuation.

        Parameters:
            None.

        Returns:
            None. The test asserts that the prepared file exists, is shorter than
            the original synthetic prompt, and has a higher RMS level.
        """
        sample_rate = 24000
        silence = np.zeros(int(sample_rate * 0.35), dtype=np.float32)
        speech = 0.02 * np.sin(
            np.linspace(0.0, np.pi * 8.0, int(sample_rate * 0.2), dtype=np.float32)
        )
        source_audio = np.concatenate([silence, speech, silence], axis=0)

        with tempfile.TemporaryDirectory() as temp_dir:
            source_audio_path = Path(temp_dir) / "voice.wav"
            prepared_audio_path = Path(temp_dir) / "prepared.wav"
            sf.write(str(source_audio_path), source_audio, sample_rate, subtype="FLOAT")

            prepared_voice = prepare_voice_conditioning(
                voice_id="voice",
                source_audio_path=source_audio_path,
                transcript="hello from a quiet prompt",
                output_audio_path=prepared_audio_path,
                target_sample_rate=sample_rate,
                preprocess_reference_audio=True,
                normalize_reference_transcript=True,
                reference_prompt_target_rms=0.1,
            )

            prepared_audio, prepared_sample_rate = sf.read(
                str(prepared_audio_path),
                dtype="float32",
            )
            prepared_file_exists = prepared_audio_path.exists()

        self.assertEqual(prepared_sample_rate, sample_rate)
        self.assertTrue(prepared_file_exists)
        self.assertTrue(prepared_voice.transcript.endswith("."))
        self.assertLess(prepared_audio.shape[0], source_audio.shape[0])
        self.assertGreater(compute_audio_rms(prepared_audio), compute_audio_rms(source_audio))
        self.assertGreater(prepared_voice.prompt_rms, prepared_voice.reference_rms)

    def test_compact_silences_preserves_configured_leading_context_for_one_region(self) -> None:
        """Ensure single-region silence compaction keeps the requested leading context.

        Usage:
            Both prompt preprocessing and generated-audio cleanup rely on silence
            compaction. This test specifically covers the one-region case that can
            easily lose preserved leading silence if slice boundaries are wrong.

        Parameters:
            None.

        Returns:
            None. The test asserts that the compacted waveform still begins with
            the configured amount of leading silence before speech samples.
        """
        sample_rate = 1000
        source_audio = np.concatenate(
            [
                np.zeros(300, dtype=np.float32),
                np.ones(100, dtype=np.float32),
                np.zeros(300, dtype=np.float32),
            ],
            axis=0,
        )

        compacted_audio = compact_silences(
            source_audio,
            sample_rate,
            threshold_db=-40.0,
            frame_milliseconds=20,
            middle_silence_milliseconds=200,
            kept_middle_silence_milliseconds=100,
            leading_silence_milliseconds=50,
            trailing_silence_milliseconds=50,
        )

        self.assertEqual(compacted_audio.shape[0], 200)
        self.assertTrue(np.allclose(compacted_audio[:50], 0.0))
        self.assertTrue(np.allclose(compacted_audio[50:150], 1.0))

    def test_clean_generated_audio_restores_original_quiet_reference_loudness(self) -> None:
        """Ensure output loudness is restored relative to the prepared prompt RMS.

        Usage:
            Quiet prompt audio may be boosted before generation to help prompt
            extraction. This test verifies that the generated audio is scaled back
            using the actual prepared-prompt RMS rather than a hard-coded target.

        Parameters:
            None.

        Returns:
            None. The test asserts that the output waveform is reduced by the
            expected reference-to-prompt RMS ratio.
        """
        generated_audio = np.full(8, 0.4, dtype=np.float32)

        cleaned_audio = clean_generated_audio(
            generated_audio,
            sample_rate=24000,
            postprocess_output_audio=False,
            reference_rms=0.02,
            prompt_rms=0.08,
        )

        self.assertTrue(np.allclose(cleaned_audio, np.full(8, 0.1, dtype=np.float32)))

    def test_merge_audio_segments_with_crossfade_inserts_configured_gap(self) -> None:
        """Ensure chunk merging preserves order and inserts a silence boundary gap.

        Usage:
            Long-form synthesis joins multiple model calls together, so this test
            protects the smoothing helper that adds a softened boundary between
            adjacent audio chunks.

        Parameters:
            None.

        Returns:
            None. The test asserts the merged length and the presence of the
            configured silence gap between the two segments.
        """
        first_segment = np.ones(100, dtype=np.float32)
        second_segment = np.full(120, 0.5, dtype=np.float32)

        merged_audio = merge_audio_segments_with_crossfade(
            [first_segment, second_segment],
            sample_rate=1000,
            crossfade_milliseconds=20,
            gap_milliseconds=50,
        )

        self.assertEqual(merged_audio.shape[0], 270)
        self.assertTrue(np.allclose(merged_audio[100:150], 0.0))
        self.assertAlmostEqual(float(merged_audio[0]), 1.0, places=5)
        self.assertAlmostEqual(float(merged_audio[-1]), 0.5, places=5)

    def test_split_audio_tail_for_crossfade_holds_only_the_requested_tail(self) -> None:
        """Ensure streaming boundary buffering keeps only the requested tail samples.

        Usage:
            Chunked streaming must hold back a small tail from each chunk so the
            following chunk can be merged smoothly. This test verifies that the
            helper divides the waveform deterministically.

        Parameters:
            None.

        Returns:
            None. The test asserts that the emitted prefix and held tail match the
            expected samples.
        """
        audio = np.arange(10, dtype=np.float32)

        emit_now_audio, held_tail_audio = split_audio_tail_for_crossfade(audio, 3)

        self.assertTrue(np.array_equal(emit_now_audio, np.arange(7, dtype=np.float32)))
        self.assertTrue(np.array_equal(held_tail_audio, np.arange(7, 10, dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
