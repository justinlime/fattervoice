"""Wyoming protocol server and event handler for Home Assistant integration."""

from __future__ import annotations

import logging
from functools import partial
from typing import Optional

from sentence_stream import SentenceBoundaryDetector
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
    SynthesizeVoice,
)

from . import __version__
from .audio import audio_to_pcm16_bytes, iter_byte_chunks
from .config import ServerConfig
from .service import SynthesisRequest, TtsService
from .voice_registry import VoiceRegistry

LOGGER = logging.getLogger(__name__)

_WYOMING_LANGUAGE_TAG_TO_MODEL_LANGUAGE = {
    "de": "German",
    "de-de": "German",
    "en": "English",
    "en-gb": "English",
    "en-us": "English",
    "es": "Spanish",
    "es-es": "Spanish",
    "fr": "French",
    "fr-fr": "French",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-hans": "Chinese",
}
_WYOMING_ADVERTISED_LANGUAGE_TAGS = tuple(_WYOMING_LANGUAGE_TAG_TO_MODEL_LANGUAGE)
_MODEL_LANGUAGE_ALIASES = {
    "auto": "Auto",
    "chinese": "Chinese",
    "english": "English",
    "french": "French",
    "german": "German",
    "spanish": "Spanish",
}



def resolve_wyoming_language(language: Optional[str]) -> Optional[str]:
    """Translate Wyoming/Home Assistant language identifiers into model language names.

    Usage:
        Wyoming clients and Home Assistant use BCP47-style language tags such as
        `en-US`, while `faster-qwen3-tts` expects human-readable language names
        like `English`. Call this helper before constructing a synthesis request
        so Home Assistant language selection remains compatible with the model.

    Parameters:
        language: The optional Wyoming/HA language identifier supplied by a client.

    Returns:
        The canonical model language name when the input can be recognized, or
        `None` when the input is empty or unknown.
    """
    if language is None:
        return None

    normalized_language = language.strip()
    if not normalized_language:
        return None

    language_key = normalized_language.replace("_", "-").lower()
    if language_key in _WYOMING_LANGUAGE_TAG_TO_MODEL_LANGUAGE:
        return _WYOMING_LANGUAGE_TAG_TO_MODEL_LANGUAGE[language_key]
    if language_key in _MODEL_LANGUAGE_ALIASES:
        return _MODEL_LANGUAGE_ALIASES[language_key]

    return None



def advertised_wyoming_languages() -> list[str]:
    """Return the Home Assistant-compatible language tags advertised for each voice.

    Usage:
        Home Assistant indexes Wyoming voices by exact advertised language tags
        and does not treat `mul` or `*` as a usable wildcard in the Assist TTS
        pipeline. This helper centralizes the explicit language tags we expose so
        voice-clone voices remain selectable in Home Assistant.

    Parameters:
        None.

    Returns:
        A stable list of Wyoming/Home Assistant language tags supported by the
        current multilingual voice-clone implementation.
    """
    return list(_WYOMING_ADVERTISED_LANGUAGE_TAGS)


class FatterQwenWyomingHandler(AsyncEventHandler):
    """Async Wyoming event handler that reuses the shared TTS synthesis service."""

    def __init__(
        self,
        service: TtsService,
        voice_registry: VoiceRegistry,
        config: ServerConfig,
        info_event: Event,
        *args,
        **kwargs,
    ) -> None:
        """Initialize a per-connection Wyoming handler.

        Usage:
            The Wyoming server factory creates one handler per client connection so
            stream state remains isolated between Home Assistant clients.

        Parameters:
            service: Shared synthesis service used for all audio generation.
            voice_registry: Shared validated voice registry.
            config: Immutable runtime configuration.
            info_event: Prebuilt Wyoming `info` event used to answer `describe`.
            *args: Positional arguments forwarded to `AsyncEventHandler`.
            **kwargs: Keyword arguments forwarded to `AsyncEventHandler`.

        Returns:
            None. A new handler instance is initialized in place.
        """
        super().__init__(*args, **kwargs)
        self.service = service
        self.voice_registry = voice_registry
        self.config = config
        self.info_event = info_event
        self._sentence_detector = SentenceBoundaryDetector()
        self._stream_voice: Optional[SynthesizeVoice] = None
        self._stream_active = False
        self._audio_started = False

    async def handle_event(self, event: Event) -> bool:
        """Route incoming Wyoming events to the correct synthesis workflow.

        Usage:
            Wyoming calls this for each decoded event on a client connection. The
            method supports standard TTS requests and streaming text-input flows.

        Parameters:
            event: The decoded Wyoming event received from the client.

        Returns:
            `True` to keep the connection open or `False` to disconnect the client.
        """
        try:
            if Describe.is_type(event.type):
                await self.write_event(self.info_event)
                return True

            if Synthesize.is_type(event.type):
                if self._stream_active:
                    LOGGER.debug(
                        "Ignoring compatibility synthesize event received during an active synthesize-start stream"
                    )
                    return True
                return await self._handle_single_synthesize(Synthesize.from_event(event))

            if SynthesizeStart.is_type(event.type):
                return await self._handle_synthesize_start(SynthesizeStart.from_event(event))

            if SynthesizeChunk.is_type(event.type):
                return await self._handle_synthesize_chunk(SynthesizeChunk.from_event(event))

            if SynthesizeStop.is_type(event.type):
                return await self._handle_synthesize_stop()

            return True
        except Exception as exc:  # pragma: no cover - exercised during runtime integration.
            LOGGER.exception("Wyoming synthesis error")
            await self.write_event(Error(text=str(exc), code=exc.__class__.__name__).event())
            self._reset_stream_state()
            return True

    async def _handle_single_synthesize(self, synthesize: Synthesize) -> bool:
        """Handle a standard non-streaming Wyoming `synthesize` request.

        Usage:
            This path is used by clients that already have the complete text. It
            performs one full synthesis request to preserve natural prosody, then
            packetizes the resulting PCM into Wyoming `audio-chunk` events.

        Parameters:
            synthesize: The parsed Wyoming synthesize request.

        Returns:
            `True` to keep the connection open after the response is sent.
        """
        synthesis_request = self._build_synthesis_request(synthesize.text, synthesize.voice)
        waveform, sample_rate = await self.service.synthesize(synthesis_request)
        pcm_payload = audio_to_pcm16_bytes(waveform)
        if not pcm_payload:
            return True

        await self.write_event(
            AudioStart(rate=sample_rate, width=2, channels=1).event()
        )
        for chunk_payload in iter_byte_chunks(
            pcm_payload,
            self.config.wyoming_audio_chunk_samples * 2,
        ):
            await self.write_event(
                AudioChunk(
                    audio=chunk_payload,
                    rate=sample_rate,
                    width=2,
                    channels=1,
                ).event()
            )
        await self.write_event(AudioStop().event())
        return True

    async def _handle_synthesize_start(self, synthesize_start: SynthesizeStart) -> bool:
        """Initialize per-connection state for Wyoming streaming text input.

        Usage:
            This is the first event in the `synthesize-start` /
            `synthesize-chunk` / `synthesize-stop` flow. It resets all sentence
            buffering state for the new stream.

        Parameters:
            synthesize_start: The parsed start event that may include the target voice.

        Returns:
            `True` to keep the connection open.
        """
        self._sentence_detector = SentenceBoundaryDetector()
        self._stream_voice = synthesize_start.voice
        self._stream_active = True
        self._audio_started = False
        return True

    async def _handle_synthesize_chunk(self, synthesize_chunk: SynthesizeChunk) -> bool:
        """Process one chunk of incoming streaming text from a Wyoming client.

        Usage:
            Completed sentences are synthesized immediately, which mirrors the
            current `wyoming-piper` sentence-boundary streaming pattern while still
            reusing the same shared synthesis service as the HTTP API.

        Parameters:
            synthesize_chunk: The parsed text chunk event.

        Returns:
            `True` to keep the connection open.
        """
        if not self._stream_active:
            raise ValueError("Received synthesize-chunk without a preceding synthesize-start event.")

        for sentence in self._sentence_detector.add_chunk(synthesize_chunk.text):
            self._audio_started = await self._emit_sentence_audio(
                sentence=sentence,
                voice=self._stream_voice,
                audio_started=self._audio_started,
            )

        return True

    async def _handle_synthesize_stop(self) -> bool:
        """Flush remaining streaming text and finish the Wyoming response handshake.

        Usage:
            This finalizes any buffered sentence fragment, sends `audio-stop` when
            audio has been emitted, and always concludes with `synthesize-stopped`.

        Parameters:
            None.

        Returns:
            `True` to keep the connection open.
        """
        if not self._stream_active:
            raise ValueError("Received synthesize-stop without a preceding synthesize-start event.")

        final_sentence = self._sentence_detector.finish()
        if final_sentence:
            self._audio_started = await self._emit_sentence_audio(
                sentence=final_sentence,
                voice=self._stream_voice,
                audio_started=self._audio_started,
            )

        if self._audio_started:
            await self.write_event(AudioStop().event())

        await self.write_event(SynthesizeStopped().event())
        self._reset_stream_state()
        return True

    async def _emit_sentence_audio(
        self,
        sentence: str,
        voice: Optional[SynthesizeVoice],
        audio_started: bool,
    ) -> bool:
        """Synthesize one completed sentence and emit Wyoming audio chunks.

        Usage:
            Only the streaming text-input Wyoming flow uses this helper. It lets
            the server start speaking on sentence boundaries without changing the
            quality-preserving full-text behavior of normal `synthesize` requests.

        Parameters:
            sentence: The finalized sentence text ready for synthesis.
            voice: The optional Wyoming voice selection metadata.
            audio_started: Whether an `audio-start` event has already been sent for
                the current logical response.

        Returns:
            A boolean indicating whether audio has been emitted after this call.
        """
        normalized_sentence = sentence.strip()
        if not normalized_sentence:
            return audio_started

        synthesis_request = self._build_synthesis_request(normalized_sentence, voice)
        bytes_per_chunk = self.config.wyoming_audio_chunk_samples * 2

        async for pcm_chunk in self.service.stream_pcm_chunks(synthesis_request):
            if not audio_started:
                await self.write_event(
                    AudioStart(
                        rate=self.service.sample_rate,
                        width=2,
                        channels=1,
                    ).event()
                )
                audio_started = True

            for chunk_payload in iter_byte_chunks(pcm_chunk, bytes_per_chunk):
                await self.write_event(
                    AudioChunk(
                        audio=chunk_payload,
                        rate=self.service.sample_rate,
                        width=2,
                        channels=1,
                    ).event()
                )

        return audio_started

    def _build_synthesis_request(
        self,
        text: str,
        voice: Optional[SynthesizeVoice],
    ) -> SynthesisRequest:
        """Translate Wyoming voice metadata into the shared synthesis request type.

        Usage:
            This helper keeps protocol-specific voice parsing out of the core TTS
            service while ensuring both protocols ultimately use the same voice
            registry and generation defaults. It also translates Home Assistant
            language tags such as `en-US` into the upstream model language names
            expected by `faster-qwen3-tts`.

        Parameters:
            text: The sentence text to synthesize.
            voice: Optional Wyoming voice metadata supplied by the client.

        Returns:
            A normalized `SynthesisRequest` suitable for the shared service.
        """
        requested_voice_id: Optional[str] = None
        requested_language: Optional[str] = None
        known_voice_ids = set(self.voice_registry.list_voice_ids())

        if voice is not None:
            if voice.language is not None:
                requested_language = resolve_wyoming_language(voice.language) or voice.language

            if voice.name:
                candidate_name = voice.name.strip()
                if candidate_name in known_voice_ids:
                    requested_voice_id = candidate_name
                elif voice.language is None:
                    requested_language = resolve_wyoming_language(candidate_name)
                    if requested_language is None:
                        requested_voice_id = candidate_name
                else:
                    requested_voice_id = candidate_name

        return SynthesisRequest(
            text=text,
            voice_id=requested_voice_id,
            language=requested_language,
        )

    def _reset_stream_state(self) -> None:
        """Reset per-connection Wyoming streaming state after a stream finishes or fails.

        Usage:
            The handler calls this helper after `synthesize-stop` and after error
            handling so stale sentence buffers do not leak into the next request.

        Parameters:
            None.

        Returns:
            None. The handler's stream-related attributes are reset in place.
        """
        self._sentence_detector = SentenceBoundaryDetector()
        self._stream_voice = None
        self._stream_active = False
        self._audio_started = False



def build_wyoming_info(voice_registry: VoiceRegistry) -> Info:
    """Build the Wyoming `info` payload advertised to discovery clients.

    Usage:
        Server startup calls this once and reuses the resulting event for every
        incoming `describe` request. The advertised voice languages are chosen to
        work with Home Assistant's exact language matching for Wyoming TTS voices.

    Parameters:
        voice_registry: The validated voice registry whose entries should be
            advertised as Wyoming voices.

    Returns:
        A fully populated Wyoming `Info` object for the TTS service.
    """
    advertised_languages = advertised_wyoming_languages()
    advertised_voices = [
        TtsVoice(
            name=voice.voice_id,
            description=(
                "Voice-clone reference built from "
                f"{voice.audio_path.name} with transcript {voice.transcript_path.name}"
            ),
            attribution=Attribution(
                name="fatterqwen",
                url="https://github.com/justinlime/fatterqwen",
            ),
            installed=True,
            version=None,
            languages=list(advertised_languages),
            speakers=None,
        )
        for voice in voice_registry.values()
    ]
    return Info(
        tts=[
            TtsProgram(
                name="fatterqwen",
                description="Wyoming TTS server backed by faster-qwen3-tts voice cloning",
                attribution=Attribution(
                    name="fatterqwen",
                    url="https://github.com/justinlime/fatterqwen",
                ),
                installed=True,
                version=__version__,
                voices=advertised_voices,
                supports_synthesize_streaming=True,
            )
        ]
    )


async def run_wyoming_server(
    service: TtsService,
    voice_registry: VoiceRegistry,
    config: ServerConfig,
) -> None:
    """Start the Wyoming server and block while it serves client connections.

    Usage:
        The main runtime starts this coroutine alongside the HTTP server when the
        user has enabled Wyoming support.

    Parameters:
        service: Shared synthesis service used by every connection handler.
        voice_registry: Shared validated voice registry.
        config: Immutable runtime configuration containing the Wyoming URI.

    Returns:
        None. The coroutine runs until the process is stopped.
    """
    if not config.wyoming_enabled or not config.wyoming_uri:
        LOGGER.info("Wyoming support is disabled")
        return

    wyoming_info = build_wyoming_info(voice_registry)
    handler_factory = partial(
        FatterQwenWyomingHandler,
        service,
        voice_registry,
        config,
        wyoming_info.event(),
    )
    LOGGER.info("Starting Wyoming server on %s", config.wyoming_uri)
    await AsyncServer.from_uri(config.wyoming_uri).run(handler_factory)
