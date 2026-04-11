# fatterqwen

`fatterqwen` is a production-oriented Python wrapper around `faster-qwen3-tts`.

It keeps the upstream CUDA-graph inference engine intact and adds:

- OpenAI-compatible `POST /v1/audio/speech`
- streaming WAV/PCM HTTP responses
- Wyoming protocol support for Home Assistant
- streamed Wyoming audio chunks and streamed Wyoming text-input handling
- a validated `voices/` directory registry based on `<voice>.<audio>` + `<voice>.txt`
- `uv`-based project management and Docker builds

## Repository layout

- `faster-qwen3-tts/`: upstream/reference inference engine
- `fatterqwen/`: wrapper package implemented in this repository
- `voices/`: runtime voice directory mounted or created by the user

## Voice directory contract

Each voice must have exactly:

- one supported reference audio file
- one matching transcript text file

Examples:

- `voices/hank.wav`
- `voices/hank.txt`
- `voices/jane.flac`
- `voices/jane.txt`

The basename becomes the public `voice` identifier exposed through both the OpenAI-compatible API and Wyoming.

## Local development with `uv`

This wrapper currently requires Python 3.11+ because the upstream `qwen-tts` dependency now resolves `onnxruntime` wheels that are no longer published for CPython 3.10.

Install dependencies:

```bash
uv sync --extra mp3 --extra dev
```

Run the server:

```bash
uv run fatterqwen \
  --voices-dir ./voices \
  --model 1.7B \
  --host 0.0.0.0 \
  --port 8000 \
  --wyoming-uri tcp://0.0.0.0:10300
```

Useful environment variables mirror the CLI flags:

- `FATTERQWEN_VOICES_DIR`
- `FATTERQWEN_HOST`
- `FATTERQWEN_PORT`
- `FATTERQWEN_MODEL`
- `FATTERQWEN_DEVICE`
- `FATTERQWEN_DTYPE`
- `FATTERQWEN_DEFAULT_LANGUAGE`
- `FATTERQWEN_CHUNK_SIZE`
- `FATTERQWEN_APPEND_SILENCE`
- `FATTERQWEN_NON_STREAMING_MODE`
- `FATTERQWEN_MAX_TEXT_LENGTH`
- `FATTERQWEN_MODEL_CACHE_DIR`
- `FATTERQWEN_PREFETCH_MANIFEST`
- `FATTERQWEN_WARMUP`
- `FATTERQWEN_WARMUP_TEXT`
- `FATTERQWEN_WYOMING_URI`
- `FATTERQWEN_WYOMING_ENABLED`
- `FATTERQWEN_WYOMING_AUDIO_CHUNK_SAMPLES`
- `FATTERQWEN_LOG_LEVEL`

## OpenAI-compatible API

Generate speech:

```bash
curl http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "tts-1",
        "input": "Hello from fatterqwen.",
        "voice": "hank",
        "response_format": "wav"
      }' \
  --output speech.wav
```

Streaming works automatically for `wav` and `pcm`. `mp3` is returned as a complete response because it must be encoded after generation.

## Wyoming support

By default the server also exposes a Wyoming TCP endpoint at `tcp://0.0.0.0:10300`.

The Wyoming handler:

- answers `describe` with `info`
- exposes the same voice registry used by HTTP
- handles `synthesize`
- handles `synthesize-start` / `synthesize-chunk` / `synthesize-stop`
- emits `audio-start` / `audio-chunk` / `audio-stop`
- buffers incoming streaming text on sentence boundaries before synthesis, mirroring the current Wyoming ecosystem pattern used by `wyoming-piper`

## Prefetching models for offline Docker/runtime use

Prefetch a model into the Hugging Face cache:

```bash
uv run fatterqwen-prefetch --model 1.7B
```

When you pass `--cache-dir`, use the actual Hugging Face hub cache directory that `from_pretrained(...)` will read from. In the Docker image, that path is `/opt/huggingface/hub`.

Prefetch both supported models:

```bash
uv run fatterqwen-prefetch --model all
```

If you want runtime to load the exact local snapshot directories directly, also write a manifest during prefetch:

```bash
uv run fatterqwen-prefetch \
  --model all \
  --cache-dir /path/to/huggingface/hub \
  --manifest /path/to/prefetched-models.json
```

## Docker

The default Docker build targets a broadly compatible CUDA runtime stack:

- base image family: official `nvidia/cuda`
- default CUDA tag: `12.6.3-cudnn-*-ubuntu24.04`
- default Python in the image: 3.12
- PyTorch wheel index: `cu126`

That keeps the stable CUDA 12.6 wheel flow from the upstream `faster-qwen3-tts` Docker reference, while moving the wrapper image to Ubuntu 24.04 so the build uses Python 3.11+ and avoids the current `onnxruntime` CPython 3.10 wheel gap.

Build the image:

```bash
docker build -t fatterqwen .
```

The Dockerfile is now single-stage and layered so that dependency installation and model prefetch happen before the frequently changing `fatterqwen/` application code is copied. That means ordinary Python logic edits should usually only invalidate the final lightweight install layer instead of forcing a full dependency and model rebuild. A `.dockerignore` file also keeps tests, voices, caches, and other non-runtime content out of the build context so Podman/Docker do less work before the first layer even starts. The build also writes `/opt/fatterqwen/prefetched-models.json`, allowing runtime startup to resolve exact local snapshot paths instead of relying only on cache discovery in offline mode.

Build an image that prefetches both supported models:

```bash
docker build \
  --build-arg MODEL_SELECTION=all \
  -t fatterqwen:all-models .
```

If you need Blackwell-oriented wheels, override the CUDA and PyTorch build arguments to a CUDA 12.8 stack:

```bash
docker build \
  --build-arg CUDA_VERSION=12.8.1 \
  --build-arg TORCH_CUDA_INDEX=cu128 \
  -t fatterqwen:cu128 .
```

Run the container:

```bash
docker run --rm --gpus all \
  -p 8000:8000 \
  -p 10300:10300 \
  -v "$PWD/voices:/opt/fatterqwen/voices:ro" \
  fatterqwen
```

## Validation

Basic lightweight validation that does not require loading a model:

```bash
python -m unittest discover -s tests
python -m compileall fatterqwen
```
