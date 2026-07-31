# fattervoice

`fattervoice` is a production-oriented Python wrapper around **OmniVoice**.

The repository name is historical, but the runtime backend is now fully OmniVoice-based. The server keeps one shared synthesis service and adds:

- OpenAI-compatible `POST /v1/audio/speech`
- chunked WAV/PCM HTTP responses
- Wyoming protocol support for Home Assistant
- streamed Wyoming text-input handling with emitted audio chunks
- a validated `voices/` directory registry based on `<voice>.<audio>` + required `<voice>.ref.txt` plus optional `<voice>.instruct.txt`
- `uv`-based project management and Docker builds
- offline-oriented model prefetch for Docker/runtime use

## Repository layout

- `fattervoice/`: wrapper package implemented in this repository
- `voices/`: runtime voice directory mounted or created by the user
- local development may temporarily include extra reference material, but the runtime depends only on the published `omnivoice` package and the code in this wrapper project

## Voice directory contract

Each voice must have exactly:

- one supported reference audio file
- one matching reference transcript text file named `<voice>.ref.txt`
- optionally, one instruct text file named `<voice>.instruct.txt`

Examples:

- `voices/hank.wav`
- `voices/hank.ref.txt`
- `voices/hank.instruct.txt` *(optional)*
- `voices/jane.flac`
- `voices/jane.ref.txt`

The basename becomes the public `voice` identifier exposed through both the OpenAI-compatible API and Wyoming.

When `<voice>.instruct.txt` is present, its stripped text is forwarded to OmniVoice as the per-voice `instruct` string for voice-design/style guidance. When it is absent, synthesis behaves the same as before.

Because `fattervoice` requires reference transcripts for every voice, OmniVoice auto-ASR is intentionally **not** part of the normal server flow. This keeps voice cloning faster, more stable, and easier to run offline.

### Supported audio formats

| Format | Extension(s) | Decoder |
| --- | --- | --- |
| WAV | `.wav` | stdlib `wave` |
| AIFF | `.aif`, `.aiff` | `soundfile` (libsndfile) |
| FLAC | `.flac` | `soundfile` (libsndfile) |
| Ogg Vorbis | `.ogg` | `soundfile` (libsndfile) |
| MP3 | `.mp3` | `pydub` + `ffmpeg` |
| Opus | `.opus` | `pydub` + `ffmpeg` |
| AAC | `.aac` | `pydub` + `ffmpeg` |
| ALAC | `.alac` | `pydub` + `ffmpeg` |

All formats are validated at startup so broken or unreadable reference files fail fast before any traffic is served. Formats decoded through `pydub` require `ffmpeg` on `PATH` (included in the official Docker image).

## Local development with `uv`

Install dependencies:

```bash
uv sync --extra mp3 --extra dev
```

Run the server:

```bash
uv run fattervoice \
  --voices-dir ./voices \
  --openapi-host 0.0.0.0 \
  --openapi-port 8000 \
  --wyoming-host 0.0.0.0 \
  --wyoming-port 10300
```

CLI arguments take precedence, and environment variables act as fallbacks.

### Server configuration reference

| CLI flag | ENV fallback | Default | Description |
| --- | --- | --- | --- |
| `--voices-dir` | `FATTERVOICE_VOICES_DIR` | `voices` | Directory containing `<voice>.<audio>` + `<voice>.ref.txt` pairs. Change when your voice files live outside the project root (e.g. a Docker volume mount). |
| `--preload-voice` | `FATTERVOICE_PRELOAD_VOICE` | *(none)* | Voice ID whose clone prompt should be built at startup instead of lazily on first request. The value must match a voice basename in `--voices-dir` (e.g. `hank` for `hank.wav` + `hank.ref.txt`). Useful when a single voice handles most traffic and you want to avoid the first-request cold-start penalty. |
| `--openapi-host` | `FATTERVOICE_OPENAPI_HOST` | `0.0.0.0` | Bind address for the OpenAI-compatible HTTP server. |
| `--openapi-port` | `FATTERVOICE_OPENAPI_PORT` | `8000` | Port for the OpenAI-compatible HTTP server. |
| `--device` | `FATTERVOICE_DEVICE` | `cuda:0` | Torch device passed to OmniVoice (e.g. `cuda:0`, `cpu`, `mps`). Use `cpu` only for debugging — GPU is required for practical latency. |
| `--dtype` | `FATTERVOICE_DTYPE` | `bfloat16` | Torch precision (`float16`, `bfloat16`, `float32`). `bfloat16` is the default balanced setting; `float16` saves VRAM but can reduce quality on some models; `float32` maximizes quality at the cost of speed and memory. |
| `--default-language` | `FATTERVOICE_DEFAULT_LANGUAGE` | `auto` | Fallback language when the client does not specify one. `auto` lets OmniVoice detect the language from the input text. Set to a concrete code (e.g. `en`, `zh`) to force a specific language. |
| `--num-step` | `FATTERVOICE_NUM_STEP` | `32` | OmniVoice diffusion decoding steps. Lower values are faster but can reduce quality; `16` is the practical minimum, `32` is the default balanced setting. |
| `--guidance-scale` | `FATTERVOICE_GUIDANCE_SCALE` | `2.0` | Classifier-free guidance scale. Higher values push output closer to the text prompt but can introduce artifacts; lower values produce more natural but less controlled speech. |
| `--denoise` / `--no-denoise` | `FATTERVOICE_DENOISE` | `true` | Enables denoise prompting for cleaner output. Disable only if you notice unwanted artifacts and want raw model output. |
| `--t-shift` | `FATTERVOICE_T_SHIFT` | `0.1` | Diffusion time-shift parameter. Adjust only if tuning diffusion scheduling for specific quality/latency trade-offs. |
| `--position-temperature` | `FATTERVOICE_POSITION_TEMPERATURE` | `5.0` | Position-sampling temperature. Higher values increase variation in prosody; lower values produce more consistent but potentially monotonous speech. |
| `--class-temperature` | `FATTERVOICE_CLASS_TEMPERATURE` | `0.0` | Class-sampling temperature. Non-zero values add stochasticity to phoneme-level choices; `0.0` (default) is deterministic. |
| `--layer-penalty-factor` | `FATTERVOICE_LAYER_PENALTY_FACTOR` | `5.0` | Penalty that biases OmniVoice toward lower codebook layers first. Higher values can improve quality by reducing reliance on higher layers; lower values allow more layer diversity. |
| `--preprocess-voice-clone-prompt` / `--no-preprocess-voice-clone-prompt` | `FATTERVOICE_PREPROCESS_VOICE_CLONE_PROMPT` | `true` | Preprocesses reference audio and transcript before caching the voice-clone prompt. Disable only if preprocessing causes issues with specific reference files. |
| `--postprocess-output-audio` / `--no-postprocess-output-audio` | `FATTERVOICE_POSTPROCESS_OUTPUT_AUDIO` | `true` | Lets OmniVoice remove excess silence and apply fade-in/fade-out to output audio. Disable if you need raw unmodified model output. |
| `--max-sentence-length` | `FATTERVOICE_MAX_SENTENCE_LENGTH` | `400` | Maximum character length of a single synthesis segment. All text is split into sentence-sized segments first; only segments exceeding this cap are broken further on word boundaries. Default 400 (~25s of speech). Lower values reduce per-segment memory at the cost of more synthesis calls. |
| `--wyoming-host` | `FATTERVOICE_WYOMING_HOST` | `0.0.0.0` | Bind address for the Wyoming TCP protocol endpoint (Home Assistant). |
| `--wyoming-port` | `FATTERVOICE_WYOMING_PORT` | `10300` | Port for the Wyoming TCP protocol endpoint. |
| `--log-level` | `FATTERVOICE_LOG_LEVEL` | `INFO` | Python logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). Use `DEBUG` for troubleshooting model loading and voice resolution. |

Boolean environment variables accept `1`, `true`, `yes`, or `on` for true, and `0`, `false`, `no`, or `off` for false.

At startup, `fattervoice` logs the fully resolved runtime configuration in a boxed summary so you can confirm the exact effective settings before it begins serving traffic.

The runtime model is now hardcoded to `omnivoice`, Wyoming support is always enabled, Wyoming audio chunking is fixed at `4096` mono PCM samples per emitted `audio-chunk`, and model-cache / prefetch-manifest selection is no longer exposed through wrapper-specific server flags.

### OmniVoice tuning defaults

The server is tuned for **fast voice cloning with strong retained quality**:

- voice-clone prompts are cached lazily on first use and retained in CPU memory so previously used voices can be reused without re-tokenizing their reference audio while inactive voices stop pinning VRAM between requests
- `num_step=32` is the default balanced speed/quality OmniVoice setting
- `dtype=bfloat16` is the default precision setting
- reference transcripts remain mandatory for stable cloning and offline operation
- OmniVoice prompt preprocessing and output postprocessing remain enabled by default

If you want to bias further toward lower latency, decrease `--num-step` to `16`.

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

All synthesis paths split text into sentence-sized segments before calling OmniVoice, so no single generation call is unbounded in memory or time. Segments that exceed `--max-sentence-length` are broken further on word boundaries.

In practice:

- **Non-streaming** (`stream=false` or `mp3`): text is split into segments, each is synthesized sequentially, and the combined waveform is returned as one response
- **Streaming** (`stream=true`, `wav`/`pcm`): PCM bytes are emitted as each sentence-sized segment completes, giving lower time-to-first-audio for long requests
- **Wyoming non-streaming** (`synthesize`): same as non-streaming above — sentence-split internally, returned as one Wyoming audio sequence
- **Wyoming streaming** (`synthesize-start` / `synthesize-chunk` / `synthesize-stop`): text arrives incrementally from the client, sentences are detected on the fly, and audio is emitted as each sentence completes
- true model-incremental audio streaming is still not claimed because OmniVoice does not currently document that capability

`mp3` is returned as a complete response because it must be encoded after generation.

## Wyoming support

By default the server also exposes a Wyoming TCP endpoint at `tcp://0.0.0.0:10300`, configured through `--wyoming-host` plus `--wyoming-port`.

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
