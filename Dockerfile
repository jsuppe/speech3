# SpeechScore API — GPU-accelerated speech analysis
# Build: docker build -t speechscore .
# Run:   docker run --gpus all -p 8000:8000 speechscore

# ── Stage 1: Base with CUDA + Python ─────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# ── Stage 2: Install Python dependencies ─────────────────────────
FROM base AS deps

WORKDIR /app

# Install pip for 3.12
RUN python3 -m ensurepip --upgrade && \
    python3 -m pip install --upgrade pip setuptools wheel

# Install Python deps (cached layer)
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

# Install spaCy model
RUN python3 -m spacy download en_core_web_sm

# ── Stage 3: Application ─────────────────────────────────────────
FROM deps AS app

WORKDIR /app

# Copy application code
COPY api/ ./api/
COPY analysis3_module.py .
COPY audio_quality.py .
COPY audio_analysis.py .
COPY advanced_audio_analysis.py .
COPY advanced_text_analysis.py .
COPY rhetorical_analysis.py .
COPY sentiment_analysis.py .
COPY voice_pipeline.py .

# Create directories
RUN mkdir -p /app/api/logs /app/api/tmp_uploads /app/data

# Model cache directory (mount as volume for persistence)
ENV HF_HOME=/app/data/huggingface \
    WHISPER_CACHE=/app/data/whisper

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/health')" || exit 1

# Expose port
EXPOSE 8000

# Run with uvicorn
CMD ["python3", "-m", "uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--log-level", "info", \
     "--timeout-keep-alive", "120"]
