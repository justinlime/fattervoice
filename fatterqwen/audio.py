"""Audio conversion helpers shared by the HTTP and Wyoming protocol adapters."""

from __future__ import annotations

import io
import struct
from typing import Iterator

import numpy as np



def audio_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    """Convert a float waveform array into little-endian 16-bit PCM bytes.

    Usage:
        Both the HTTP streaming endpoint and the Wyoming adapter use this helper
        to turn model output chunks into transport-friendly PCM payloads.

    Parameters:
        audio: A NumPy array containing mono waveform samples in the typical
        `[-1.0, 1.0]` floating-point range.

    Returns:
        A byte string containing signed 16-bit PCM samples.
    """
    clipped_audio = np.clip(audio, -1.0, 1.0)
    return (clipped_audio * 32767.0).astype(np.int16).tobytes()



def build_wav_header(sample_rate: int, data_size: int = 0xFFFFFFFF) -> bytes:
    """Build a RIFF/WAV header for complete or streaming audio responses.

    Usage:
        HTTP handlers prepend this header when serving WAV output. Supplying the
        default `data_size` value produces a streaming-friendly header whose final
        payload length is intentionally left unknown.

    Parameters:
        sample_rate: The waveform sampling rate in Hertz.
        data_size: The PCM payload size in bytes, or `0xFFFFFFFF` for streaming.

    Returns:
        The serialized WAV header bytes.
    """
    channels = 1
    sample_width_bits = 16
    byte_rate = sample_rate * channels * sample_width_bits // 8
    block_align = channels * sample_width_bits // 8
    riff_size = 0xFFFFFFFF if data_size == 0xFFFFFFFF else 36 + data_size

    buffer = io.BytesIO()
    buffer.write(b"RIFF")
    buffer.write(struct.pack("<I", riff_size))
    buffer.write(b"WAVE")
    buffer.write(b"fmt ")
    buffer.write(
        struct.pack(
            "<IHHIIHH",
            16,
            1,
            channels,
            sample_rate,
            byte_rate,
            block_align,
            sample_width_bits,
        )
    )
    buffer.write(b"data")
    buffer.write(struct.pack("<I", data_size))
    return buffer.getvalue()



def build_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert a waveform array into a complete in-memory WAV file.

    Usage:
        Non-streaming HTTP responses call this helper when the client asks for a
        standard WAV file instead of raw PCM or MP3.

    Parameters:
        audio: The waveform samples to encode.
        sample_rate: The sampling rate in Hertz.

    Returns:
        A byte string containing a complete WAV file.
    """
    pcm_bytes = audio_to_pcm16_bytes(audio)
    return build_wav_header(sample_rate=sample_rate, data_size=len(pcm_bytes)) + pcm_bytes



def encode_mp3_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode a waveform array into MP3 bytes using pydub/ffmpeg.

    Usage:
        The OpenAI-compatible HTTP endpoint uses this helper for `response_format`
        values of `mp3`. The conversion is intentionally full-buffer because the
        current server design streams WAV/PCM directly and leaves MP3 as a
        post-generation encoding step.

    Parameters:
        audio: The waveform samples to encode.
        sample_rate: The sampling rate in Hertz.

    Returns:
        A byte string containing a complete MP3 file.

    Raises:
        RuntimeError: If `pydub` is not installed in the runtime environment.
    """
    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise RuntimeError(
            "MP3 output requires the optional `pydub` dependency and ffmpeg in the runtime image."
        ) from exc

    segment = AudioSegment(
        data=audio_to_pcm16_bytes(audio),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )
    output_buffer = io.BytesIO()
    segment.export(output_buffer, format="mp3")
    return output_buffer.getvalue()



def iter_byte_chunks(payload: bytes, chunk_size: int) -> Iterator[bytes]:
    """Yield fixed-size slices from a byte payload without copying more than necessary.

    Usage:
        The Wyoming adapter uses this helper to repacketize model-sized PCM output
        chunks into smaller Wyoming `audio-chunk` payloads.

    Parameters:
        payload: The full byte payload that should be split.
        chunk_size: The maximum size of each yielded byte chunk.

    Returns:
        An iterator that yields byte slices in order until the payload is exhausted.
    """
    for start_index in range(0, len(payload), chunk_size):
        yield payload[start_index : start_index + chunk_size]
