ARG MODEL_SELECTION=all

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04
ARG MODEL_SELECTION
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

# Keep the expensive dependency install and model-prefetch layers tied only to
# dependency metadata and the small prefetch helper surface.
COPY pyproject.toml uv.lock /app/
COPY faster-qwen3-tts/pyproject.toml /app/faster-qwen3-tts/pyproject.toml
COPY faster-qwen3-tts/README.md /app/faster-qwen3-tts/README.md
COPY faster-qwen3-tts/faster_qwen3_tts /app/faster-qwen3-tts/faster_qwen3_tts
COPY fatterqwen/__init__.py /app/fatterqwen/__init__.py
COPY fatterqwen/hf_cache.py /app/fatterqwen/hf_cache.py
COPY fatterqwen/model_catalog.py /app/fatterqwen/model_catalog.py
COPY fatterqwen/prefetch.py /app/fatterqwen/prefetch.py
COPY fatterqwen/prefetch_manifest.py /app/fatterqwen/prefetch_manifest.py

RUN uv sync --frozen --extra mp3 --no-install-project

# Prefetch the selected models during the image build so runtime stays offline.
RUN mkdir -p /opt/fatterqwen/voices /opt/torchinductor && \
    uv run --no-sync python -m fatterqwen.prefetch \
    --model "${MODEL_SELECTION}" \
    --cache-dir "${HF_HUB_CACHE}" \
    --manifest /opt/fatterqwen/prefetched-models.json

COPY README.md /app/README.md
COPY fatterqwen /app/fatterqwen
COPY docker-entrypoint.sh /app/docker-entrypoint.sh

RUN uv pip install --python /opt/venv/bin/python --no-deps /app && \
    chmod +x /app/docker-entrypoint.sh

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
