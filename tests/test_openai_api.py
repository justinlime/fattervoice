"""Unit tests for the OpenAI-compatible HTTP adapter."""

from __future__ import annotations

import unittest
from unittest.mock import Mock

from fastapi.testclient import TestClient

from fattervoice.openai_api import create_openai_app


class FailingStreamingService:
    """Minimal service double that fails streamed requests before iteration starts.

    Usage:
        The HTTP adapter tests inject this class so they can verify that
        `create_openai_app(...)` converts eager streaming validation failures into
        normal HTTP 400 responses instead of opening a chunked body first.

    Parameters:
        None.

    Returns:
        None. The instance exposes only the attributes and methods used by the
        OpenAI-compatible adapter.
    """

    model_id = "k2-fsa/OmniVoice"
    sample_rate = 24000

    def stream_pcm_chunks(self, request):
        """Raise a validation-style error for the buffered chunked streaming path.

        Usage:
            The adapter calls this method for WAV/PCM chunked responses when the
            request does not explicitly opt into the lower-latency streaming mode.

        Parameters:
            request: The normalized synthesis request created by the adapter.

        Returns:
            This method never returns because it always raises `ValueError`.
        """
        _ = request
        raise ValueError("text invalid")

    def stream_low_latency_pcm_chunks(self, request):
        """Raise a validation-style error for explicit low-latency streaming requests.

        Usage:
            The adapter calls this method when `stream=true` is set for WAV/PCM
            responses, so the test double raises immediately to verify the HTTP
            layer still returns a regular 400 response before any bytes are sent.

        Parameters:
            request: The normalized synthesis request created by the adapter.

        Returns:
            This method never returns because it always raises `ValueError`.
        """
        _ = request
        raise ValueError("text invalid")

    async def synthesize(self, request):
        """Fail fast if a non-streaming code path reaches this test double unexpectedly.

        Usage:
            The streaming validation test should never call buffered synthesis, so
            this method raises an assertion if the adapter uses the wrong path.

        Parameters:
            request: The normalized synthesis request created by the adapter.

        Returns:
            This method never returns because it always raises `AssertionError`.
        """
        _ = request
        raise AssertionError("Buffered synthesis should not be used in this test.")


class RoutingStreamingService:
    """Minimal service double that records which HTTP streaming path was selected.

    Usage:
        The OpenAI adapter now distinguishes between the default buffered chunked
        WAV/PCM path and the explicit low-latency `stream=true` path. Tests use
        this double to verify that routing decision without loading the real TTS
        stack.

    Parameters:
        None.

    Returns:
        None. The instance records which streaming helper the adapter called.
    """

    model_id = "k2-fsa/OmniVoice"
    sample_rate = 24000

    def __init__(self) -> None:
        """Initialize call tracking for the lightweight routing test double.

        Usage:
            Each test creates a fresh instance so the called streaming method can
            be asserted without interference from prior requests.

        Parameters:
            None.

        Returns:
            None. The instance stores an empty `calls` list.
        """
        self.calls: list[str] = []

    def stream_pcm_chunks(self, request):
        """Record use of the default buffered chunked streaming path.

        Usage:
            The OpenAI adapter calls this method for WAV/PCM chunked responses
            when low-latency sentence-segmented streaming was not explicitly
            requested by the client.

        Parameters:
            request: The normalized synthesis request created by the adapter.

        Returns:
            An async iterator that yields one small PCM chunk.
        """
        _ = request
        self.calls.append("buffered")

        async def emit_chunks():
            """Yield one deterministic PCM chunk for adapter-level tests.

            Usage:
                This nested generator keeps the fake service compatible with the
                production adapter's expectation of an async chunk iterator.

            Parameters:
                None. It closes over the recorded call state.

            Returns:
                An async iterator that yields one PCM chunk.
            """
            yield b"\x00\x00"

        return emit_chunks()

    def stream_low_latency_pcm_chunks(self, request):
        """Record use of the explicit low-latency sentence-segmented path.

        Usage:
            The OpenAI adapter calls this method when the HTTP request sets
            `stream=true` for a WAV/PCM response.

        Parameters:
            request: The normalized synthesis request created by the adapter.

        Returns:
            An async iterator that yields one small PCM chunk.
        """
        _ = request
        self.calls.append("low_latency")

        async def emit_chunks():
            """Yield one deterministic PCM chunk for adapter-level tests.

            Usage:
                This nested generator keeps the fake service compatible with the
                production adapter's expectation of an async chunk iterator.

            Parameters:
                None. It closes over the recorded call state.

            Returns:
                An async iterator that yields one PCM chunk.
            """
            yield b"\x00\x00"

        return emit_chunks()

    async def synthesize(self, request):
        """Fail fast if buffered full synthesis is used during a routing test.

        Usage:
            The routing tests are only about streamed WAV/PCM selection, so this
            method guards against accidental use of the wrong adapter branch.

        Parameters:
            request: The normalized synthesis request created by the adapter.

        Returns:
            This method never returns because it always raises `AssertionError`.
        """
        _ = request
        raise AssertionError("Buffered synthesis should not be used in this test.")


class MinimalVoiceRegistry:
    """Small registry double that satisfies the adapter's discovery endpoints.

    Usage:
        The OpenAI adapter exposes health and voice-list routes alongside the
        speech endpoint, so tests provide this registry instead of the full voice
        scanner implementation.

    Parameters:
        None.

    Returns:
        None. The instance exposes the minimal registry surface used by the app.
    """

    def list_voice_ids(self) -> list[str]:
        """Return one predictable voice ID for lightweight adapter tests.

        Usage:
            Health-route responses call this method to enumerate known voices.

        Parameters:
            None.

        Returns:
            A one-item list containing the synthetic `demo` voice ID.
        """
        return ["demo"]

    def values(self) -> list[object]:
        """Return an empty voice metadata list for routes that enumerate entries.

        Usage:
            The adapter's optional voice-list route iterates over this method's
            return value. The streaming validation test does not depend on any
            actual metadata, so an empty list is sufficient.

        Parameters:
            None.

        Returns:
            An empty list of voice entries.
        """
        return []


class OpenAiApiTests(unittest.TestCase):
    """Verify request/error handling in the OpenAI-compatible speech endpoint."""

    def test_explicit_stream_true_uses_low_latency_streaming_path(self) -> None:
        """Ensure explicit `stream=true` selects sentence-segmented low-latency streaming.

        Usage:
            This regression test protects the HTTP adapter routing that now sends
            explicit streaming requests through the lower-latency sequential
            sentence-synthesis helper instead of the default buffered chunked path.

        Parameters:
            None.

        Returns:
            None. The test asserts that the expected fake streaming method was
            called and that the endpoint still returns HTTP 200.
        """
        service = RoutingStreamingService()
        app = create_openai_app(service, MinimalVoiceRegistry(), Mock())
        client = TestClient(app)

        response = client.post(
            "/v1/audio/speech",
            json={
                "input": "Hello from fattervoice.",
                "voice": "demo",
                "response_format": "wav",
                "stream": True,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(service.calls, ["low_latency"])

    def test_streaming_validation_failure_returns_http_400(self) -> None:
        """Ensure eager streaming failures become normal HTTP 400 responses.

        Usage:
            This regression test covers the adapter path that now calls the
            selected streaming helper before returning `StreamingResponse`, which
            is required to surface validation failures before any bytes are sent.

        Parameters:
            None.

        Returns:
            None. The test asserts that the endpoint responds with status 400 and
            the validation detail from the service.
        """
        app = create_openai_app(FailingStreamingService(), MinimalVoiceRegistry(), Mock())
        client = TestClient(app)

        response = client.post(
            "/v1/audio/speech",
            json={
                "input": "   ",
                "voice": "demo",
                "response_format": "wav",
                "stream": True,
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "text invalid"})


if __name__ == "__main__":
    unittest.main()
