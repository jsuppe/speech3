# SpeakFit GPU Worker
# Runs Whisper transcription jobs from Redis queue

FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-gpu.txt .
RUN pip install --no-cache-dir -r requirements-gpu.txt

# Pre-download Whisper model during build (faster startup)
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='int8')"

# Copy worker code
COPY worker/ ./worker/
COPY api/transcription.py ./api/
COPY api/confidence_analysis.py ./api/

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Health endpoint
EXPOSE 8080

CMD ["celery", "-A", "worker.tasks", "worker", "--loglevel=info", "--concurrency=2"]
