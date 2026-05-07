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
COPY fattervoice/__init__.py /app/fattervoice/__init__.py
COPY fattervoice/model_catalog.py /app/fattervoice/model_catalog.py
COPY fattervoice/prefetch.py /app/fattervoice/prefetch.py

RUN uv sync --extra mp3 --no-install-project

# Prefetch the selected OmniVoice assets during the image build so runtime stays offline.
RUN mkdir -p /opt/fattervoice/voices /opt/torchinductor && \
    uv run --no-sync python -m fattervoice.prefetch \
    --model "${MODEL_SELECTION}"

COPY README.md /app/README.md
COPY fattervoice /app/fattervoice
COPY docker-entrypoint.sh /app/docker-entrypoint.sh

RUN uv pip install --python /opt/venv/bin/python --no-deps /app && \
    chmod +x /app/docker-entrypoint.sh

ENV FATTERVOICE_VOICES_DIR=/opt/fattervoice/voices \
    FATTERVOICE_HOST=0.0.0.0 \
    FATTERVOICE_PORT=8000 \
    FATTERVOICE_WYOMING_ENABLED=true \
    FATTERVOICE_WYOMING_URI=tcp://0.0.0.0:10300 \
    MODEL_SELECTION_HINT=${MODEL_SELECTION} \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8000 10300
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["fattervoice"]
