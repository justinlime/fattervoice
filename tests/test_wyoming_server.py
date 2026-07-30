"""Unit tests for the Wyoming protocol adapter."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
import wave
from pathlib import Path

from fattervoice.voice_registry import VoiceRegistry

try:
    from wyoming.info import Describe
    from wyoming.tts import SynthesizeChunk, SynthesizeStart, SynthesizeStop, SynthesizeVoice

    from fattervoice.wyoming_server import (
        FatterVoiceWyomingHandler,
        advertised_wyoming_languages,
        build_wyoming_info,
    )
except ModuleNotFoundError as import_error:  # pragma: no cover - dependency availability varies by environment.
    WYOMING_IMPORT_ERROR = import_error
else:
    WYOMING_IMPORT_ERROR = None


if WYOMING_IMPORT_ERROR is None:

    class FakeStreamingService:
        """Minimal streaming service double used to verify Wyoming event handling.

        Usage:
            Wyoming handler tests inject this class instead of the real GPU-backed
            synthesis service so they can assert on request translation and emitted
            audio events without loading a model.

        Parameters:
            pcm_chunks: Optional ordered list of PCM byte chunks that should be
                yielded for each streaming request.

        Returns:
            None. The instance records every received synthesis request.
        """

        def __init__(self, pcm_chunks: list[bytes] | None = None) -> None:
            self.sample_rate = 24000
            self.pcm_chunks = pcm_chunks or [b"\x00\x01", b"\x02\x03"]
            self.requests = []

        async def stream_pcm_chunks(self, request):
            """Yield predetermined PCM chunks while recording the normalized request.

            Usage:
                Streaming Wyoming tests call the handler against this method to
                confirm that sentence-boundary synthesis emits the expected protocol
                events and request fields.

            Parameters:
                request: The normalized `SynthesisRequest` supplied by the handler.

            Returns:
                An async iterator of PCM byte chunks.
            """
            self.requests.append(request)
            for pcm_chunk in self.pcm_chunks:
                yield pcm_chunk


    class RecordingWyomingHandler(FatterVoiceWyomingHandler):
        """Wyoming handler test double that records outgoing events in memory.

        Usage:
            Tests subclass the production handler so they can inspect the exact event
            types emitted for discovery and streaming flows without opening sockets.

        Parameters:
            The parameters are identical to `FatterVoiceWyomingHandler`.

        Returns:
            None. The handler stores emitted events on `recorded_events`.
        """

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, reader=asyncio.StreamReader(), writer=object(), **kwargs)
            self.recorded_events = []

        async def write_event(self, event) -> None:
            """Capture outgoing Wyoming events instead of writing to a socket.

            Usage:
                Tests call normal handler methods and then inspect `recorded_events`
                to verify Wyoming protocol behavior.

            Parameters:
                event: The Wyoming event that would normally be written to the client.

            Returns:
                None. The event is appended to `recorded_events`.
            """
            self.recorded_events.append(event)


    class WyomingServerTests(unittest.IsolatedAsyncioTestCase):
        """Verify Home Assistant compatibility and streaming event behavior."""

        def _create_voice_registry(self) -> VoiceRegistry:
            """Create a temporary one-voice registry for Wyoming adapter tests.

            Usage:
                Each test calls this helper to build a valid registry with one voice
                named `hank`, matching the project contract of audio/transcript pairs.

            Parameters:
                None.

            Returns:
                A validated `VoiceRegistry` backed by a temporary directory that will
                be cleaned up automatically after the test completes.
            """
            temporary_directory = tempfile.TemporaryDirectory()
            self.addCleanup(temporary_directory.cleanup)
            voices_dir = Path(temporary_directory.name)
            audio_path = voices_dir / "hank.wav"
            transcript_path = voices_dir / "hank.ref.txt"

            with wave.open(str(audio_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(b"\x00\x00" * 160)

            transcript_path.write_text("Hello from Hank.", encoding="utf-8")
            return VoiceRegistry.scan(voices_dir)

        def _create_handler(self, service: FakeStreamingService | None = None) -> RecordingWyomingHandler:
            """Construct a recording Wyoming handler with lightweight fake dependencies.

            Usage:
                Tests call this helper to exercise the production handler logic while
                controlling the voice registry, Wyoming info payload, and emitted PCM.

            Parameters:
                service: Optional fake synthesis service to inject. When omitted, a
                    default `FakeStreamingService` is created.

            Returns:
                A ready-to-use `RecordingWyomingHandler` instance.
            """
            voice_registry = self._create_voice_registry()
            fake_service = service or FakeStreamingService()
            return RecordingWyomingHandler(
                fake_service,
                voice_registry,
                build_wyoming_info(voice_registry).event(),
            )

        async def test_handle_event_describe_emits_info_response(self) -> None:
            """Ensure `describe` requests receive the required Wyoming `info` event.

            Usage:
                This test protects Home Assistant discovery by verifying that the
                handler implements the required `describe` -> `info` handshake.

            Parameters:
                None.

            Returns:
                None. The test asserts on the emitted event type.
            """
            handler = self._create_handler()

            keep_connection_open = await handler.handle_event(Describe().event())

            self.assertTrue(keep_connection_open)
            self.assertEqual([event.type for event in handler.recorded_events], ["info"])

        async def test_streaming_flow_emits_audio_stop_then_synthesize_stopped(self) -> None:
            """Ensure streaming Wyoming synthesis follows the documented response order.

            Usage:
                This test verifies the `synthesize-start` / `synthesize-chunk` /
                `synthesize-stop` flow used by Home Assistant when a Wyoming TTS
                service advertises streaming support.

            Parameters:
                None.

            Returns:
                None. The test asserts on the emitted event sequence and normalized
                synthesis request.
            """
            fake_service = FakeStreamingService()
            handler = self._create_handler(service=fake_service)

            await handler.handle_event(SynthesizeStart(voice=SynthesizeVoice(name="hank")).event())
            await handler.handle_event(SynthesizeChunk(text="Hello world.").event())
            await handler.handle_event(SynthesizeStop().event())

            self.assertEqual(
                [event.type for event in handler.recorded_events],
                ["audio-start", "audio-chunk", "audio-chunk", "audio-stop", "synthesize-stopped"],
            )
            self.assertEqual(len(fake_service.requests), 1)
            self.assertEqual(fake_service.requests[0].voice_id, "hank")
            self.assertIsNone(fake_service.requests[0].language)

        async def test_build_synthesis_request_maps_home_assistant_language_tags(self) -> None:
            """Ensure Home Assistant language tags are translated for OmniVoice.

            Usage:
                Wyoming/HA language selection uses BCP47-like tags such as `en-US`,
                while OmniVoice prefers compact language IDs such as `en`. This
                test verifies the adapter performs that translation before inference.

            Parameters:
                None.

            Returns:
                None. The test asserts on the normalized synthesis request fields.
            """
            handler = self._create_handler()

            synthesis_request = handler._build_synthesis_request(
                text="Hello there",
                voice=SynthesizeVoice(name="hank", language="en-US"),
            )

            self.assertEqual(synthesis_request.voice_id, "hank")
            self.assertEqual(synthesis_request.language, "en")

        async def test_build_synthesis_request_handles_language_only_voice_payloads(self) -> None:
            """Ensure language-only Wyoming voice payloads still resolve to a model language.

            Usage:
                Some Wyoming client paths represent a language-only selection through
                the `name` field instead of an explicit `language` field. This test
                verifies the adapter still treats known language tags as languages
                rather than as invalid voice identifiers.

            Parameters:
                None.

            Returns:
                None. The test asserts that the resulting request has no voice ID and
                uses the expected OmniVoice language ID.
            """
            handler = self._create_handler()

            synthesis_request = handler._build_synthesis_request(
                text="Bonjour",
                voice=SynthesizeVoice(name="fr-FR"),
            )

            self.assertIsNone(synthesis_request.voice_id)
            self.assertEqual(synthesis_request.language, "fr")

        async def test_build_wyoming_info_uses_home_assistant_compatible_languages(self) -> None:
            """Ensure advertised Wyoming voices use explicit HA-compatible language tags.

            Usage:
                Home Assistant indexes Wyoming voices by exact language tag and does
                not treat `mul` as a wildcard. This test protects the advertised info
                payload so discovered voices remain selectable in Assist.

            Parameters:
                None.

            Returns:
                None. The test asserts that the voice advertises explicit language
                tags and no longer uses `mul`.
            """
            voice_registry = self._create_voice_registry()

            info = build_wyoming_info(voice_registry)
            advertised_voice = info.tts[0].voices[0]

            self.assertEqual(advertised_voice.name, "hank")
            self.assertNotIn("mul", advertised_voice.languages)
            self.assertEqual(advertised_voice.languages, advertised_wyoming_languages())

else:

    @unittest.skip(f"Wyoming test dependencies are unavailable: {WYOMING_IMPORT_ERROR}")
    class WyomingServerTests(unittest.TestCase):
        """Placeholder test case used when Wyoming dependencies are unavailable."""

        def test_wyoming_dependencies_available(self) -> None:
            """Document why Wyoming adapter tests were skipped in this environment.

            Usage:
                This placeholder keeps unittest discovery stable in lightweight
                environments that do not have the Wyoming runtime dependencies
                installed outside the project virtual environment.

            Parameters:
                None.

            Returns:
                None. The test is always skipped by the enclosing class decorator.
            """
            self.fail("This test should be skipped when Wyoming dependencies are missing.")


if __name__ == "__main__":
    unittest.main()
