"""FastAPI application factory for the OpenAI-compatible TTS HTTP interface."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from .audio import audio_to_pcm16_bytes, build_wav_bytes, build_wav_header, encode_mp3_bytes
from .config import ServerConfig
from .service import SynthesisRequest, TtsService
from .voice_registry import VoiceRegistry, VoiceRegistryError

LOGGER = logging.getLogger(__name__)


class SpeechRequest(BaseModel):
    """OpenAI-compatible request body for `POST /v1/audio/speech`."""

    model: str = Field(default="tts-1")
    input: str
    voice: Optional[str] = None
    response_format: str = Field(default="wav")
    speed: float = Field(default=1.0)
    stream: Optional[bool] = None
    language: Optional[str] = None



def create_openai_app(service: TtsService, voice_registry: VoiceRegistry, config: ServerConfig) -> FastAPI:
    """Create the HTTP application that exposes the OpenAI-compatible TTS routes.

    Usage:
        Startup code calls this once after the shared `TtsService` is ready. The
        resulting FastAPI application can then be handed to Uvicorn.

    Parameters:
        service: The shared synthesis service used for all audio generation.
        voice_registry: The validated voice registry exposed to clients.
        config: Immutable server configuration needed by request validation.

    Returns:
        A configured FastAPI application.
    """
    app = FastAPI(title="fatterqwen OpenAI-compatible TTS API")

    @app.get("/health")
    async def health() -> dict[str, object]:
        """Report basic liveness and the currently loaded model metadata.

        Usage:
            This lightweight route is intended for Docker health checks and simple
            operational monitoring.

        Parameters:
            None.

        Returns:
            A JSON-serializable dictionary describing the process health.
        """
        return {
            "status": "ok",
            "model": service.model_id,
            "sample_rate": service.sample_rate,
            "voices": voice_registry.list_voice_ids(),
        }

    @app.get("/v1/models")
    async def list_models() -> dict[str, object]:
        """Expose the currently loaded TTS model through an OpenAI-style model list.

        Usage:
            Some OpenAI-compatible clients expect to discover available models via
            this route before sending a synthesis request.

        Parameters:
            None.

        Returns:
            An OpenAI-style list payload containing the active model.
        """
        return {
            "object": "list",
            "data": [
                {
                    "id": service.model_id,
                    "object": "model",
                    "owned_by": "fatterqwen",
                }
            ],
        }

    @app.get("/v1/audio/voices")
    async def list_voices() -> dict[str, object]:
        """Return the canonical voice registry used by every transport adapter.

        Usage:
            This route is not part of the official OpenAI API, but it is useful
            for operators and client-side tooling that need to enumerate available
            voice IDs before making synthesis requests.

        Parameters:
            None.

        Returns:
            A JSON object containing voice IDs and source file metadata.
        """
        return {
            "data": [
                {
                    "id": voice.voice_id,
                    "audio_path": str(voice.audio_path),
                    "transcript_path": str(voice.transcript_path),
                }
                for voice in voice_registry.values()
            ]
        }

    @app.post("/v1/audio/speech")
    async def create_speech(request: SpeechRequest):
        """Generate speech from text using the configured shared synthesis service.

        Usage:
            This route mirrors the common OpenAI TTS contract. WAV and PCM can be
            streamed as chunked responses; MP3 is returned after full encoding.

        Parameters:
            request: The validated HTTP request body parsed by FastAPI/Pydantic.

        Returns:
            Either a `StreamingResponse` for streamed audio or a regular `Response`
            for fully buffered output formats.

        Raises:
            HTTPException: If the request is invalid or synthesis fails.
        """
        response_format = request.response_format.lower()
        content_types = {
            "wav": "audio/wav",
            "pcm": "audio/pcm",
            "mp3": "audio/mpeg",
        }
        if response_format not in content_types:
            raise HTTPException(
                status_code=400,
                detail="response_format must be one of: wav, pcm, mp3",
            )

        if request.speed != 1.0:
            LOGGER.warning(
                "Ignoring unsupported speed=%s request parameter because faster-qwen3-tts does not expose speed control.",
                request.speed,
            )

        synthesis_request = SynthesisRequest(
            text=request.input,
            voice_id=request.voice,
            language=request.language,
        )

        try:
            service.validate_request(synthesis_request)
        except VoiceRegistryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        stream_response = (
            response_format in {"wav", "pcm"}
            if request.stream is None
            else request.stream
        )

        if stream_response and response_format in {"wav", "pcm"}:

            async def streamed_audio_body():
                """Yield a streaming HTTP audio body using the shared synthesis service.

                Usage:
                    This nested generator feeds FastAPI's `StreamingResponse` with
                    a WAV header (when needed) followed by model-produced PCM chunks.

                Parameters:
                    None. It closes over the validated request state.

                Returns:
                    An async byte stream suitable for chunked HTTP transfer.
                """
                try:
                    if response_format == "wav":
                        yield build_wav_header(service.sample_rate)
                    async for pcm_chunk in service.stream_pcm_chunks(synthesis_request):
                        yield pcm_chunk
                except Exception:  # pragma: no cover - exercised during runtime integration.
                    LOGGER.exception("Unhandled streaming synthesis failure")
                    raise

            return StreamingResponse(
                streamed_audio_body(),
                media_type=content_types[response_format],
            )

        try:
            waveform, sample_rate = await service.synthesize(synthesis_request)
            if response_format == "wav":
                payload = build_wav_bytes(waveform, sample_rate)
            elif response_format == "pcm":
                payload = audio_to_pcm16_bytes(waveform)
            else:
                payload = encode_mp3_bytes(waveform, sample_rate)
        except VoiceRegistryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            if response_format == "mp3":
                raise HTTPException(status_code=501, detail=str(exc)) from exc
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - exercised in runtime integration.
            LOGGER.exception("Unhandled synthesis failure")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return Response(content=payload, media_type=content_types[response_format])

    return app
