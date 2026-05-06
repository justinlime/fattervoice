ARG CUDA_VERSION=12.6.3
ARG UBUNTU_VERSION=24.04
ARG MODEL_SELECTION=all
ARG TORCH_CUDA_INDEX=cu126

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION}
ARG MODEL_SELECTION
ARG TORCH_CUDA_INDEX
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:/bin:${PATH}" \
    HF_HOME=/opt/huggingface \
    HF_HUB_CACHE=/opt/huggingface/hub \
    HUGGINGFACE_HUB_CACHE=/opt/huggingface/hub \
    TRANSFORMERS_CACHE=/opt/huggingface/hub \
    TORCHINDUCTOR_CACHE_DIR=/opt/torchinductor

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    ffmpeg \
    git \
    libsndfile1 \
    python3 \
    python3-venv \
    sox \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only dependency metadata, the local faster-qwen3-tts dependency, and the
# small subset of fatterqwen modules required for prefetching. This keeps the
# expensive dependency-install and model-prefetch layers reusable when only the
# main server logic changes.
COPY pyproject.toml /app/
COPY faster-qwen3-tts/pyproject.toml /app/faster-qwen3-tts/pyproject.toml
COPY faster-qwen3-tts/README.md /app/faster-qwen3-tts/README.md
COPY faster-qwen3-tts/faster_qwen3_tts /app/faster-qwen3-tts/faster_qwen3_tts
COPY fatterqwen/__init__.py /app/fatterqwen/__init__.py
COPY fatterqwen/hf_cache.py /app/fatterqwen/hf_cache.py
COPY fatterqwen/model_catalog.py /app/fatterqwen/model_catalog.py
COPY fatterqwen/prefetch.py /app/fatterqwen/prefetch.py
COPY fatterqwen/prefetch_manifest.py /app/fatterqwen/prefetch_manifest.py

RUN python3 - <<'PY'
from pathlib import Path
import os
pyproject_path = Path('/app/pyproject.toml')
updated_content = pyproject_path.read_text(encoding='utf-8').replace(
    'https://download.pytorch.org/whl/cu126',
    f"https://download.pytorch.org/whl/{os.environ['TORCH_CUDA_INDEX']}",
)
pyproject_path.write_text(updated_content, encoding='utf-8')
PY

RUN uv sync --extra mp3 --no-install-project

# Prefetch models before copying the main application source so this costly layer
# stays cached across ordinary Python logic edits. The manifest records the exact
# local snapshot paths so runtime can load those directories directly in offline
# containers.
RUN mkdir -p /opt/fatterqwen /opt/fatterqwen/voices /opt/torchinductor
RUN uv run --no-sync python -m fatterqwen.prefetch \
    --model "${MODEL_SELECTION}" \
    --cache-dir "${HF_HUB_CACHE}" \
    --manifest /opt/fatterqwen/prefetched-models.json

# Copy the frequently changing application code last so simple logic changes only
# invalidate the lightweight final install layer instead of dependency/model layers.
COPY README.md /app/README.md
COPY fatterqwen /app/fatterqwen
COPY docker-entrypoint.sh /app/docker-entrypoint.sh

RUN uv pip install --python /opt/venv/bin/python --no-deps /app
RUN chmod +x /app/docker-entrypoint.sh

ENV FATTERQWEN_VOICES_DIR=/opt/fatterqwen/voices \
    FATTERQWEN_HOST=0.0.0.0 \
    FATTERQWEN_PORT=8000 \
    FATTERQWEN_WYOMING_ENABLED=true \
    FATTERQWEN_WYOMING_URI=tcp://0.0.0.0:10300 \
    FATTERQWEN_MODEL_CACHE_DIR=/opt/huggingface/hub \
    FATTERQWEN_PREFETCH_MANIFEST=/opt/fatterqwen/prefetched-models.json \
    MODEL_SELECTION_HINT=${MODEL_SELECTION} \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8000 10300
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["fatterqwen"]
