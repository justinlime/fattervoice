"""Unit tests for the OpenAI-compatible HTTP adapter."""

from __future__ import annotations

import unittest
from unittest.mock import Mock

from fastapi.testclient import TestClient

from fatterqwen.openai_api import create_openai_app


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

    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    sample_rate = 24000

    def stream_pcm_chunks(self, request):
        """Raise a validation-style error as soon as the streaming path is requested.

        Usage:
            The adapter calls this method before returning `StreamingResponse` so
            bad requests can still become regular HTTP errors.

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

    def test_streaming_validation_failure_returns_http_400(self) -> None:
        """Ensure eager streaming failures become normal HTTP 400 responses.

        Usage:
            This regression test covers the adapter path that now calls
            `stream_pcm_chunks(...)` before returning `StreamingResponse`, which
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
