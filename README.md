# fattervoice

`fattervoice` is a production-oriented Python wrapper around **OmniVoice**.

The repository name is historical, but the runtime backend is now fully OmniVoice-based. The server keeps one shared synthesis service and adds:

- OpenAI-compatible `POST /v1/audio/speech`
- chunked WAV/PCM HTTP responses
- Wyoming protocol support for Home Assistant
- streamed Wyoming text-input handling with emitted audio chunks
- a validated `voices/` directory registry based on `<voice>.<audio>` + `<voice>.txt`
- `uv`-based project management and Docker builds
- offline-oriented model prefetch for Docker/runtime use

## Repository layout

- `fattervoice/`: wrapper package implemented in this repository
- `voices/`: runtime voice directory mounted or created by the user
- local development may temporarily include extra reference material, but the runtime depends only on the published `omnivoice` package and the code in this wrapper project

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

Because `fattervoice` requires transcripts for every voice, OmniVoice auto-ASR is intentionally **not** part of the normal server flow. This keeps voice cloning faster, more stable, and easier to run offline.

## Local development with `uv`

Install dependencies:

```bash
uv sync --extra mp3 --extra dev
```

Run the server:

```bash
uv run fattervoice \
  --voices-dir ./voices \
  --model omnivoice \
  --host 0.0.0.0 \
  --port 8000 \
  --wyoming-uri tcp://0.0.0.0:10300
```

CLI arguments take precedence, and environment variables act as fallbacks.

### Server configuration reference

| CLI flag | ENV fallback | Default |
| --- | --- | --- |
| `--voices-dir` | `FATTERVOICE_VOICES_DIR` | `voices` |
| `--host` | `FATTERVOICE_HOST` | `0.0.0.0` |
| `--port` | `FATTERVOICE_PORT` | `8000` |
| `--model` | `FATTERVOICE_MODEL` | `omnivoice` |
| `--device` | `FATTERVOICE_DEVICE` | `cuda:0` |
| `--dtype` | `FATTERVOICE_DTYPE` | `float16` |
| `--default-language` | `FATTERVOICE_DEFAULT_LANGUAGE` | `auto` |
| `--max-text-length` | `FATTERVOICE_MAX_TEXT_LENGTH` | `4000` |
| `--num-step` | `FATTERVOICE_NUM_STEP` | `16` |
| `--guidance-scale` | `FATTERVOICE_GUIDANCE_SCALE` | `2.0` |
| `--denoise` / `--no-denoise` | `FATTERVOICE_DENOISE` | `true` |
| `--t-shift` | `FATTERVOICE_T_SHIFT` | `0.1` |
| `--position-temperature` | `FATTERVOICE_POSITION_TEMPERATURE` | `5.0` |
| `--class-temperature` | `FATTERVOICE_CLASS_TEMPERATURE` | `0.0` |
| `--layer-penalty-factor` | `FATTERVOICE_LAYER_PENALTY_FACTOR` | `5.0` |
| `--preprocess-voice-clone-prompt` / `--no-preprocess-voice-clone-prompt` | `FATTERVOICE_PREPROCESS_VOICE_CLONE_PROMPT` | `true` |
| `--postprocess-output-audio` / `--no-postprocess-output-audio` | `FATTERVOICE_POSTPROCESS_OUTPUT_AUDIO` | `true` |
| `--audio-chunk-duration` | `FATTERVOICE_AUDIO_CHUNK_DURATION` | `15.0` |
| `--audio-chunk-threshold` | `FATTERVOICE_AUDIO_CHUNK_THRESHOLD` | `30.0` |
| `--wyoming-enabled` / `--no-wyoming-enabled` | `FATTERVOICE_WYOMING_ENABLED` | `true` |
| `--wyoming-uri` | `FATTERVOICE_WYOMING_URI` | `tcp://0.0.0.0:10300` |
| `--log-level` | `FATTERVOICE_LOG_LEVEL` | `INFO` |

Boolean environment variables accept `1`, `true`, `yes`, or `on` for true, and `0`, `false`, `no`, or `off` for false.

Wyoming audio chunking is now fixed at `4096` mono PCM samples per emitted `audio-chunk`, and model-cache / prefetch-manifest selection is no longer exposed through wrapper-specific server flags.

### OmniVoice tuning defaults

The server is tuned for **fast voice cloning with strong retained quality**:

- voice-clone prompts are cached lazily on first use and retained in CPU memory so previously used voices can be reused without re-tokenizing their reference audio while inactive voices stop pinning VRAM between requests
- `num_step=16` is the default speed-focused OmniVoice setting
- reference transcripts remain mandatory for stable cloning and offline operation
- OmniVoice prompt preprocessing and output postprocessing remain enabled by default

If you want to bias further toward quality, increase `--num-step` to `32`.

## OpenAI-compatible API

Generate speech:

```bash
curl http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "tts-1",
        "input": "Hello from fattervoice.",
        "voice": "hank",
        "response_format": "wav",
        "speed": 1.0
      }' \
  --output speech.wav
```

`speed` is forwarded to OmniVoice.

### Streaming behavior

`wav` and `pcm` responses are still available as chunked HTTP responses, but OmniVoice currently exposes **buffered full-audio generation** rather than a documented true incremental audio streaming API. In practice that means:

- default chunked WAV/PCM responses still stream bytes after each full buffered OmniVoice generation call completes
- explicit `stream=true` on WAV/PCM switches to a lower-latency sentence-segmented path that synthesizes one sentence-like segment at a time to improve time-to-first-audio for longer requests
- Wyoming sentence-streaming still works by synthesizing completed text chunks
- true model-incremental audio streaming is still not claimed because the backend does not currently document that capability

`mp3` is returned as a complete response because it must be encoded after generation.

## Wyoming support

By default the server also exposes a Wyoming TCP endpoint at `tcp://0.0.0.0:10300`.

The Wyoming handler:

- answers `describe` with `info`
- exposes the same voice registry used by HTTP
- handles `synthesize`
- handles `synthesize-start` / `synthesize-chunk` / `synthesize-stop`
- emits `audio-start` / `audio-chunk` / `audio-stop`
- buffers incoming streaming text on sentence boundaries before synthesis

## Prefetching models for offline Docker/runtime use

Prefetch the built-in OmniVoice model into the Hugging Face cache:

```bash
uv run fattervoice-prefetch --model omnivoice
```

This command also prefetches the auxiliary Higgs audio tokenizer repository required by OmniVoice when it is not already embedded in the resolved snapshot layout.

### Prefetch command reference

| CLI flag | ENV fallback | Default |
| --- | --- | --- |
| `--model` | — | `omnivoice` |

The prefetch helper now relies on the active Hugging Face cache configuration instead of wrapper-specific cache or manifest flags. In the Docker image, that cache path is hardcoded to `/opt/huggingface/hub`.

## Docker

The default Docker build uses the current CUDA 12.6-based `uv` workflow from this repository while prefetching OmniVoice assets so runtime can stay offline.

Build the image:

```bash
docker build -t fattervoice .
```

Because the built-in runtime currently targets one primary OmniVoice model alias, both of the following build styles are valid and equivalent for the default backend:

```bash
docker build --build-arg MODEL_SELECTION=omnivoice -t fattervoice:omnivoice .
docker build --build-arg MODEL_SELECTION=all -t fattervoice:all .
```

Run the container:

```bash
docker run --rm --gpus all \
  -p 8000:8000 \
  -p 10300:10300 \
  -v "$PWD/voices:/opt/fattervoice/voices:ro" \
  fattervoice
```

## Validation

Basic lightweight validation that does not require loading a model:

```bash
uv run python -m unittest discover -s tests
uv run python -m compileall fattervoice tests
```
