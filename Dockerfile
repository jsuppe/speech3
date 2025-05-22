# Use a smaller base image
FROM python:3.9-slim

WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies with specific versions to ensure compatibility
RUN pip install --no-cache-dir pip==21.3.1 && \
    pip install --no-cache-dir huggingface_hub==0.4.0 && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m nltk.downloader punkt && \
    python -m spacy download en_core_web_sm

# Copy Python scripts
COPY analysis3.py analysis3-json.py setup.py ./
COPY precache_models.py ./

# Create a models directory and precache the models
RUN mkdir -p /app/models /app/output && \
    chmod 777 /app/models /app/output && \
    # Pre-download models to avoid downloading at runtime
    python precache_models.py

# Set environment variables for model caching
ENV SENTENCE_TRANSFORMERS_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HOME=/app/models
ENV RESULTS_FILE_PATH=/app/output/results.txt

# Default entrypoint is the original script
ENTRYPOINT ["python", "analysis3.py"]

