"""Standalone quality helpers for reference preparation, long-form chunking, and audio cleanup.

This module intentionally contains self-owned implementations inspired by common
TTS quality practices so `fattervoice` does not depend on external example
repositories at runtime.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

_SENTENCE_BOUNDARIES = set(".!?;:\n。！？；：")
_CLOSING_MARKS = set('"”’\')]}》」』】')
_END_PUNCTUATION = set(".!?;:,。！？；：，、…")
_REFERENCE_SILENCE_THRESHOLD_DB = -45.0
_OUTPUT_SILENCE_THRESHOLD_DB = -45.0
_FRAME_ANALYSIS_MILLISECONDS = 20
_REFERENCE_INTERNAL_SILENCE_MILLISECONDS = 250
_REFERENCE_LEADING_SILENCE_MILLISECONDS = 100
_REFERENCE_TRAILING_SILENCE_MILLISECONDS = 200
_REFERENCE_KEPT_GAP_MILLISECONDS = 160
_OUTPUT_INTERNAL_SILENCE_MILLISECONDS = 500
_OUTPUT_LEADING_SILENCE_MILLISECONDS = 100
_OUTPUT_TRAILING_SILENCE_MILLISECONDS = 100
_OUTPUT_KEPT_GAP_MILLISECONDS = 180


@dataclass(frozen=True)
class PreparedVoiceConditioning:
    """Prepared voice-cloning assets derived from a registry voice entry.

    Usage:
        The service caches one prepared conditioning object per voice so request
        handling can reuse cleaned prompt audio and normalized transcript text.

    Parameters:
        voice_id: The public voice identifier associated with the prepared data.
        audio_path: The local path to the processed prompt audio file.
        transcript: The transcript text that should accompany the processed audio.
        reference_rms: The RMS level measured from the original reference audio.
        prompt_rms: The RMS level of the processed audio actually written to disk.

    Returns:
        A frozen container describing the prompt assets ready for synthesis.
    """

    voice_id: str
    audio_path: Path
    transcript: str
    reference_rms: float
    prompt_rms: float


def ensure_terminal_punctuation(text: str) -> str:
    """Ensure that prompt or transcript text ends with a sentence-closing mark.

    Usage:
        Reference transcripts often come from plain text files without trailing
        punctuation, which can make TTS prosody less stable. This helper adds a
        language-appropriate terminal marker only when one is missing.

    Parameters:
        text: The raw transcript or request text that should be normalized.

    Returns:
        The original text with trailing whitespace removed and a final
        punctuation mark added when needed.
    """
    normalized_text = text.strip()
    if not normalized_text:
        return normalized_text

    trailing_closing_marks = ""
    core_text = normalized_text
    while core_text and core_text[-1] in _CLOSING_MARKS:
        trailing_closing_marks = core_text[-1] + trailing_closing_marks
        core_text = core_text[:-1]

    if not core_text:
        return normalized_text
    if core_text[-1] in _END_PUNCTUATION:
        return core_text + trailing_closing_marks

    contains_cjk = any("\u4e00" <= character <= "\u9fff" for character in core_text)
    return core_text + ("。" if contains_cjk else ".") + trailing_closing_marks


def _character_speech_weight(character: str) -> float:
    """Estimate the relative speaking cost of one character for chunk planning.

    Usage:
        Long-form chunking needs a language-aware length heuristic that is more
        stable than plain character count. This helper assigns lightweight
        phonetic weights based on Unicode category and script ranges.

    Parameters:
        character: A single Unicode character from the synthesis text.

    Returns:
        A floating-point weight representing the approximate speech effort for
        that character.
    """
    code_point = ord(character)
    category = unicodedata.category(character)

    if category.startswith("Z"):
        return 0.2
    if category.startswith("P") or category.startswith("S"):
        return 0.5
    if category.startswith("N"):
        return 2.5
    if category.startswith("M"):
        return 0.0

    if 0x4E00 <= code_point <= 0x9FFF or 0x3400 <= code_point <= 0x4DBF:
        return 2.8
    if 0x3040 <= code_point <= 0x30FF:
        return 2.1
    if 0xAC00 <= code_point <= 0xD7AF or 0x1100 <= code_point <= 0x11FF:
        return 2.3
    if 0x0590 <= code_point <= 0x08FF:
        return 1.4
    if 0x0900 <= code_point <= 0x0D7F or 0x0E00 <= code_point <= 0x0EFF:
        return 1.7
    if 0x0400 <= code_point <= 0x052F:
        return 1.0
    if 0x0370 <= code_point <= 0x03FF:
        return 1.0

    return 1.0


def estimate_text_speech_units(text: str) -> float:
    """Estimate the speaking weight of a text string for long-form chunking.

    Usage:
        The service or future long-form helpers can use this heuristic to decide
        when a request should be split into multiple synthesis chunks when a
        backend-specific duration estimator is unavailable.

    Parameters:
        text: The synthesis text whose approximate spoken size should be measured.

    Returns:
        A floating-point total weight where larger values represent longer or
        denser spoken content.
    """
    return sum(_character_speech_weight(character) for character in text.strip())


def _split_text_into_sentences(text: str) -> list[str]:
    """Split text into sentence-like fragments while keeping boundary punctuation.

    Usage:
        Long-form chunk planning prefers splitting at natural punctuation marks so
        each model call receives coherent sub-sentences instead of arbitrary raw
        slices.

    Parameters:
        text: The normalized synthesis text to break into sentence fragments.

    Returns:
        A list of non-empty sentence-like strings in their original order.
    """
    sentences: list[str] = []
    current_characters: list[str] = []
    normalized_text = text.strip()

    for character in normalized_text:
        if character == "\r":
            continue
        if not current_characters and sentences and character in _CLOSING_MARKS:
            sentences[-1] += character
            continue

        current_characters.append(character)
        if character in _SENTENCE_BOUNDARIES:
            sentence = "".join(current_characters).strip()
            if sentence:
                sentences.append(sentence)
            current_characters = []

    trailing_sentence = "".join(current_characters).strip()
    if trailing_sentence:
        sentences.append(trailing_sentence)

    return sentences or [normalized_text]


def _find_soft_split_index(text: str, target_units: float) -> int:
    """Find a soft split point near the target weight inside one long sentence.

    Usage:
        When an individual sentence is too large to fit comfortably inside one
        synthesis chunk, this helper searches backward for a whitespace or
        punctuation boundary before falling back to a hard character cut.

    Parameters:
        text: The oversized sentence that must be split.
        target_units: The preferred approximate speech weight for one chunk.

    Returns:
        The character index where the sentence should be divided.
    """
    accumulated_units = 0.0
    best_index = 0

    for index, character in enumerate(text, start=1):
        accumulated_units += _character_speech_weight(character)
        if character.isspace() or character in _SENTENCE_BOUNDARIES or character == ",":
            best_index = index
        if accumulated_units >= target_units:
            if best_index > 0:
                return best_index
            return index

    return len(text)


def _split_overlong_sentence(text: str, target_units: float) -> list[str]:
    """Split one oversized sentence into smaller synthesis-friendly fragments.

    Usage:
        Sentence-level chunking can still produce fragments that are too long for
        stable voice cloning. This helper recursively breaks such fragments down
        using soft split points near the configured target weight.

    Parameters:
        text: The oversized sentence that should be subdivided.
        target_units: The approximate desired speech weight for each output part.

    Returns:
        A list of smaller non-empty fragments that preserve the original text
        order as closely as possible.
    """
    remaining_text = text.strip()
    if not remaining_text:
        return []

    fragments: list[str] = []
    while estimate_text_speech_units(remaining_text) > target_units and len(remaining_text) > 1:
        split_index = _find_soft_split_index(remaining_text, target_units)
        left_fragment = remaining_text[:split_index].strip()
        if not left_fragment:
            break
        fragments.append(left_fragment)
        remaining_text = remaining_text[split_index:].strip()

    if remaining_text:
        fragments.append(remaining_text)

    return fragments


def split_text_for_longform_synthesis(
    text: str,
    *,
    threshold_units: int,
    target_units: int,
    min_units: int,
) -> list[str]:
    """Split long text into natural chunks for more stable multi-call synthesis.

    Usage:
        The service uses this helper when a request is large enough that one
        monolithic model call is more likely to drift in pacing or consistency.
        Short requests are returned unchanged as a one-item list.

    Parameters:
        text: The raw synthesis text to examine.
        threshold_units: The minimum estimated speech weight that activates
            chunking at all.
        target_units: The preferred approximate weight of each chunk once
            chunking is activated.
        min_units: The minimum approximate weight of a finished chunk before a
            tiny trailing fragment should be merged back into its predecessor.

    Returns:
        A list of one or more ordered chunks ready to pass individually into the
        TTS model.
    """
    normalized_text = text.strip()
    if not normalized_text:
        return []

    total_units = estimate_text_speech_units(normalized_text)
    if total_units <= float(threshold_units):
        return [normalized_text]

    sentence_candidates: list[str] = []
    for sentence in _split_text_into_sentences(normalized_text):
        sentence_candidates.extend(_split_overlong_sentence(sentence, float(target_units)))

    chunks: list[str] = []
    current_chunk = ""
    current_units = 0.0

    for sentence in sentence_candidates:
        sentence_units = estimate_text_speech_units(sentence)
        if not current_chunk:
            current_chunk = sentence
            current_units = sentence_units
            continue

        if current_units + sentence_units <= float(target_units):
            separator = "" if current_chunk.endswith("\n") else " "
            current_chunk = f"{current_chunk}{separator}{sentence}".strip()
            current_units += sentence_units
            continue

        chunks.append(current_chunk.strip())
        current_chunk = sentence
        current_units = sentence_units

    if current_chunk:
        chunks.append(current_chunk.strip())

    if len(chunks) >= 2 and estimate_text_speech_units(chunks[-1]) < float(min_units):
        chunks[-2] = f"{chunks[-2]} {chunks[-1]}".strip()
        chunks.pop()

    return [chunk for chunk in chunks if chunk]


def compute_audio_rms(audio: np.ndarray) -> float:
    """Measure the root-mean-square level of a mono waveform.

    Usage:
        Reference RMS is captured before prompt normalization so generated audio
        can optionally be returned closer to the original loudness profile.

    Parameters:
        audio: A mono waveform represented as a one-dimensional float array.

    Returns:
        The RMS level as a non-negative floating-point value.
    """
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))


def load_audio_mono(audio_path: Path, target_sample_rate: int) -> np.ndarray:
    """Load an audio file, mix it to mono, and resample it for prompt preparation.

    Usage:
        Reference audio files in the voice registry may arrive in different
        formats or sample rates. This helper converts them into one consistent
        mono float32 representation for preprocessing and caching.

    Parameters:
        audio_path: The source audio file to read from disk.
        target_sample_rate: The sample rate expected by the active TTS model.

    Returns:
        A one-dimensional mono float32 waveform at the requested sample rate.
    """
    waveform, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    mono_waveform = waveform.mean(axis=1)

    if sample_rate != target_sample_rate:
        resampled_waveform = torchaudio.functional.resample(
            torch.from_numpy(mono_waveform).unsqueeze(0),
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        )
        mono_waveform = resampled_waveform.squeeze(0).cpu().numpy()

    return mono_waveform.astype(np.float32, copy=False)


def detect_nonsilent_ranges(
    audio: np.ndarray,
    sample_rate: int,
    *,
    threshold_db: float,
    frame_milliseconds: int,
    min_silence_milliseconds: int,
) -> list[tuple[int, int]]:
    """Detect contiguous non-silent regions inside a mono waveform.

    Usage:
        Both reference preprocessing and output cleanup need a lightweight silence
        detector without relying on optional external audio tooling. This helper
        performs frame-wise RMS analysis and returns sample ranges that contain
        meaningful speech or audible content.

    Parameters:
        audio: The mono waveform to inspect.
        sample_rate: The waveform sample rate in Hertz.
        threshold_db: Frames quieter than this dB value are treated as silence.
        frame_milliseconds: The analysis frame size in milliseconds.
        min_silence_milliseconds: Gaps shorter than this threshold are merged so
            brief pauses do not split one utterance into many fragments.

    Returns:
        A list of `(start_sample, end_sample)` ranges covering the detected
        non-silent regions in ascending order.
    """
    if audio.size == 0:
        return []

    frame_samples = max(1, int(sample_rate * frame_milliseconds / 1000))
    merge_gap_samples = max(0, int(sample_rate * min_silence_milliseconds / 1000))
    frame_ranges: list[tuple[int, int]] = []

    for frame_start in range(0, int(audio.size), frame_samples):
        frame_end = min(int(audio.size), frame_start + frame_samples)
        frame = audio[frame_start:frame_end]
        frame_rms = compute_audio_rms(frame)
        frame_db = -120.0 if frame_rms <= 1e-12 else float(20.0 * math.log10(frame_rms))
        if frame_db > threshold_db:
            frame_ranges.append((frame_start, frame_end))

    if not frame_ranges:
        if np.max(np.abs(audio)) > 1e-6:
            return [(0, int(audio.size))]
        return []

    merged_ranges: list[tuple[int, int]] = [frame_ranges[0]]
    for next_start, next_end in frame_ranges[1:]:
        current_start, current_end = merged_ranges[-1]
        if next_start - current_end <= merge_gap_samples:
            merged_ranges[-1] = (current_start, next_end)
        else:
            merged_ranges.append((next_start, next_end))

    return merged_ranges


def compact_silences(
    audio: np.ndarray,
    sample_rate: int,
    *,
    threshold_db: float,
    frame_milliseconds: int,
    middle_silence_milliseconds: int,
    kept_middle_silence_milliseconds: int,
    leading_silence_milliseconds: int,
    trailing_silence_milliseconds: int,
) -> np.ndarray:
    """Remove or shrink long silences while preserving short natural pauses.

    Usage:
        This helper is shared by prompt preprocessing and generated-audio cleanup.
        It keeps a small amount of contextual silence at the edges and between
        non-silent regions so speech remains natural while dead air is reduced.

    Parameters:
        audio: The mono waveform to compact.
        sample_rate: The waveform sample rate in Hertz.
        threshold_db: Frames quieter than this dB value are treated as silence.
        frame_milliseconds: The RMS analysis frame size in milliseconds.
        middle_silence_milliseconds: Gaps at least this long are considered
            removable long silences.
        kept_middle_silence_milliseconds: The amount of silence to preserve for
            each removable interior gap.
        leading_silence_milliseconds: The maximum leading silence to keep before
            the first non-silent region.
        trailing_silence_milliseconds: The maximum trailing silence to keep after
            the last non-silent region.

    Returns:
        A new mono waveform with long silences compacted. The return value may be
        empty if the input contains no meaningful non-silent content.
    """
    nonsilent_ranges = detect_nonsilent_ranges(
        audio,
        sample_rate,
        threshold_db=threshold_db,
        frame_milliseconds=frame_milliseconds,
        min_silence_milliseconds=middle_silence_milliseconds,
    )
    if not nonsilent_ranges:
        return np.zeros(0, dtype=np.float32)

    leading_keep_samples = max(0, int(sample_rate * leading_silence_milliseconds / 1000))
    trailing_keep_samples = max(0, int(sample_rate * trailing_silence_milliseconds / 1000))
    kept_gap_samples = max(0, int(sample_rate * kept_middle_silence_milliseconds / 1000))
    preserved_segments: list[np.ndarray] = []
    last_range_index = len(nonsilent_ranges) - 1

    for range_index, (range_start, range_end) in enumerate(nonsilent_ranges):
        if range_index > 0:
            previous_end = nonsilent_ranges[range_index - 1][1]
            gap_length = max(0, range_start - previous_end)
            if gap_length > 0 and kept_gap_samples > 0:
                left_keep = min(gap_length // 2, kept_gap_samples // 2)
                right_keep = min(gap_length - left_keep, kept_gap_samples - left_keep)
                if left_keep > 0:
                    preserved_segments.append(audio[previous_end:previous_end + left_keep])
                if right_keep > 0:
                    preserved_segments.append(audio[range_start - right_keep:range_start])

        segment_start = (
            max(0, range_start - leading_keep_samples)
            if range_index == 0
            else range_start
        )
        segment_end = (
            min(int(audio.size), range_end + trailing_keep_samples)
            if range_index == last_range_index
            else range_end
        )
        preserved_segments.append(audio[segment_start:segment_end])

    compacted_audio = np.concatenate(preserved_segments, axis=0)
    return compacted_audio.astype(np.float32, copy=False)


def scale_audio_to_target_rms(audio: np.ndarray, target_rms: float) -> tuple[np.ndarray, float]:
    """Boost a quiet waveform toward a target RMS without changing louder clips.

    Usage:
        Prompt extraction tends to benefit from reference audio that is not too
        quiet. This helper raises very quiet inputs toward a configurable target
        level while leaving normal-loudness recordings untouched.

    Parameters:
        audio: The mono waveform to potentially rescale.
        target_rms: The desired minimum RMS level for prepared prompt audio.

    Returns:
        A tuple of `(scaled_audio, scaled_rms)` where `scaled_audio` is the
        processed waveform and `scaled_rms` is its new RMS value.
    """
    current_rms = compute_audio_rms(audio)
    if audio.size == 0 or target_rms <= 0.0 or current_rms <= 0.0 or current_rms >= target_rms:
        return audio.astype(np.float32, copy=False), current_rms

    scaled_audio = np.clip(audio * (target_rms / current_rms), -1.0, 1.0)
    return scaled_audio.astype(np.float32, copy=False), compute_audio_rms(scaled_audio)


def prepare_voice_conditioning(
    *,
    voice_id: str,
    source_audio_path: Path,
    transcript: str,
    output_audio_path: Path,
    target_sample_rate: int,
    preprocess_reference_audio: bool,
    normalize_reference_transcript: bool,
    reference_prompt_target_rms: float,
) -> PreparedVoiceConditioning:
    """Create cached prompt assets for one voice-cloning reference pair.

    Usage:
        The service calls this once per voice and caches the result so repeated
        requests can reuse a pre-cleaned prompt waveform and transcript without
        reprocessing the original files each time.

    Parameters:
        voice_id: The public voice identifier being prepared.
        source_audio_path: The original reference audio file from the voice
            registry.
        transcript: The transcript text paired with the reference audio.
        output_audio_path: The destination path where the processed prompt audio
            should be written.
        target_sample_rate: The sample rate expected by the active TTS model.
        preprocess_reference_audio: Whether silence compaction should be applied
            before the prompt file is cached.
        normalize_reference_transcript: Whether missing terminal punctuation
            should be added to the transcript.
        reference_prompt_target_rms: The minimum RMS level for cached prompt
            audio after quiet-reference normalization.

    Returns:
        A `PreparedVoiceConditioning` object describing the cached prompt assets.
    """
    prepared_transcript = (
        ensure_terminal_punctuation(transcript)
        if normalize_reference_transcript
        else transcript.strip()
    )
    prepared_audio = load_audio_mono(source_audio_path, target_sample_rate)
    reference_rms = compute_audio_rms(prepared_audio)

    if preprocess_reference_audio:
        compacted_audio = compact_silences(
            prepared_audio,
            target_sample_rate,
            threshold_db=_REFERENCE_SILENCE_THRESHOLD_DB,
            frame_milliseconds=_FRAME_ANALYSIS_MILLISECONDS,
            middle_silence_milliseconds=_REFERENCE_INTERNAL_SILENCE_MILLISECONDS,
            kept_middle_silence_milliseconds=_REFERENCE_KEPT_GAP_MILLISECONDS,
            leading_silence_milliseconds=_REFERENCE_LEADING_SILENCE_MILLISECONDS,
            trailing_silence_milliseconds=_REFERENCE_TRAILING_SILENCE_MILLISECONDS,
        )
        if compacted_audio.size > 0:
            prepared_audio = compacted_audio

    prepared_audio, prompt_rms = scale_audio_to_target_rms(
        prepared_audio,
        reference_prompt_target_rms,
    )
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(output_audio_path),
        prepared_audio,
        target_sample_rate,
        subtype="FLOAT",
    )

    return PreparedVoiceConditioning(
        voice_id=voice_id,
        audio_path=output_audio_path,
        transcript=prepared_transcript,
        reference_rms=reference_rms,
        prompt_rms=prompt_rms,
    )


def clean_generated_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    postprocess_output_audio: bool,
    reference_rms: float,
    prompt_rms: float,
) -> np.ndarray:
    """Clean a generated waveform before final boundary finishing is applied.

    Usage:
        This helper performs the content-preserving parts of output cleanup that
        should happen on individual chunk waveforms before they are merged or
        streamed, such as long-silence compaction and loudness restoration.

    Parameters:
        audio: The generated mono waveform to clean.
        sample_rate: The waveform sample rate in Hertz.
        postprocess_output_audio: Whether silence cleanup should be applied.
        reference_rms: The RMS level of the original reference audio.
        prompt_rms: The RMS level of the prepared prompt audio actually used for
            generation, allowing quiet-prompt gain to be reversed accurately.

    Returns:
        A cleaned mono float32 waveform with no final fade or padding applied.
    """
    cleaned_audio = np.asarray(audio, dtype=np.float32).flatten()

    if postprocess_output_audio:
        compacted_audio = compact_silences(
            cleaned_audio,
            sample_rate,
            threshold_db=_OUTPUT_SILENCE_THRESHOLD_DB,
            frame_milliseconds=_FRAME_ANALYSIS_MILLISECONDS,
            middle_silence_milliseconds=_OUTPUT_INTERNAL_SILENCE_MILLISECONDS,
            kept_middle_silence_milliseconds=_OUTPUT_KEPT_GAP_MILLISECONDS,
            leading_silence_milliseconds=_OUTPUT_LEADING_SILENCE_MILLISECONDS,
            trailing_silence_milliseconds=_OUTPUT_TRAILING_SILENCE_MILLISECONDS,
        )
        if compacted_audio.size > 0:
            cleaned_audio = compacted_audio

    if 0.0 < reference_rms < prompt_rms:
        cleaned_audio = cleaned_audio * (reference_rms / prompt_rms)

    return np.clip(cleaned_audio, -1.0, 1.0).astype(np.float32, copy=False)


def apply_output_fade_and_padding(
    audio: np.ndarray,
    sample_rate: int,
    *,
    fade_milliseconds: int,
    pad_milliseconds: int,
) -> np.ndarray:
    """Apply small fade and silence padding to the edges of a finished waveform.

    Usage:
        Final output should not start or end abruptly after chunk merging or
        silence compaction. This helper adds gentle boundary treatment once the
        waveform is otherwise complete.

    Parameters:
        audio: The mono waveform that should receive final edge finishing.
        sample_rate: The waveform sample rate in Hertz.
        fade_milliseconds: The fade-in/out duration in milliseconds.
        pad_milliseconds: The silence padding duration to add on each side.

    Returns:
        A new mono waveform with edge fades and optional silence padding.
    """
    finished_audio = np.asarray(audio, dtype=np.float32).flatten().copy()
    if finished_audio.size == 0:
        return finished_audio

    fade_samples = max(0, int(sample_rate * fade_milliseconds / 1000))
    if fade_samples > 0:
        fade_length = min(fade_samples, finished_audio.size // 2)
        if fade_length > 0:
            fade_in = np.linspace(0.0, 1.0, fade_length, dtype=np.float32)
            fade_out = np.linspace(1.0, 0.0, fade_length, dtype=np.float32)
            finished_audio[:fade_length] *= fade_in
            finished_audio[-fade_length:] *= fade_out

    pad_samples = max(0, int(sample_rate * pad_milliseconds / 1000))
    if pad_samples > 0:
        silence = np.zeros(pad_samples, dtype=np.float32)
        finished_audio = np.concatenate([silence, finished_audio, silence], axis=0)

    return finished_audio.astype(np.float32, copy=False)


def merge_audio_segments_with_crossfade(
    audio_segments: list[np.ndarray],
    sample_rate: int,
    *,
    crossfade_milliseconds: int,
    gap_milliseconds: int,
) -> np.ndarray:
    """Join chunk waveforms with softened boundaries for long-form synthesis.

    Usage:
        When one request is synthesized as multiple model calls, this helper adds
        a short faded boundary and an optional gap between chunk waveforms so the
        final result sounds less abrupt than a hard concat.

    Parameters:
        audio_segments: Ordered mono waveforms that should be combined.
        sample_rate: The waveform sample rate in Hertz.
        crossfade_milliseconds: The duration of the fade-out/fade-in region at
            each chunk boundary.
        gap_milliseconds: The amount of silence to insert between adjacent chunks.

    Returns:
        One merged mono waveform containing all input segments in order.
    """
    if not audio_segments:
        return np.zeros(0, dtype=np.float32)

    merged_audio = np.asarray(audio_segments[0], dtype=np.float32).flatten().copy()
    gap_samples = max(0, int(sample_rate * gap_milliseconds / 1000))
    crossfade_samples = max(0, int(sample_rate * crossfade_milliseconds / 1000))
    gap_audio = np.zeros(gap_samples, dtype=np.float32)

    for next_segment in audio_segments[1:]:
        next_audio = np.asarray(next_segment, dtype=np.float32).flatten().copy()
        boundary_audio = merged_audio.copy()
        fade_length = min(crossfade_samples, boundary_audio.size, next_audio.size)
        if fade_length > 0:
            fade_out = np.linspace(1.0, 0.0, fade_length, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, fade_length, dtype=np.float32)
            boundary_audio[-fade_length:] *= fade_out
            next_audio[:fade_length] *= fade_in
        merged_audio = np.concatenate([boundary_audio, gap_audio, next_audio], axis=0)

    return merged_audio.astype(np.float32, copy=False)


def split_audio_tail_for_crossfade(audio: np.ndarray, tail_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Split a waveform into an emit-now prefix and a hold-back tail buffer.

    Usage:
        Long-form streaming cannot know how to smooth the end of one chunk until
        the next chunk arrives. This helper withholds a short tail that can later
        be merged into the following audio segment.

    Parameters:
        audio: The mono waveform that should be divided.
        tail_samples: How many trailing samples to hold back for a future join.

    Returns:
        A tuple of `(emit_now_audio, held_tail_audio)` where the first element can
        be streamed immediately and the second element is reserved for the next
        boundary merge.
    """
    normalized_audio = np.asarray(audio, dtype=np.float32).flatten()
    if tail_samples <= 0 or normalized_audio.size <= tail_samples:
        return np.zeros(0, dtype=np.float32), normalized_audio.copy()
    return normalized_audio[:-tail_samples].copy(), normalized_audio[-tail_samples:].copy()
