# SpeechScore API — Deployment Guide

## Option A: Docker (Recommended)

### Prerequisites
- Docker 24+
- NVIDIA Container Toolkit (`nvidia-ctk`)
- NVIDIA GPU with 6+ GB VRAM (8+ GB recommended)

### Quick Start

```bash
# Build and run
docker compose up -d

# Check logs
docker compose logs -f speechscore

# Verify
curl http://localhost:8000/v1/health
```

### First Run
On first startup, the API will:
1. Download Whisper large-v3 (~3 GB) to the model cache volume
2. Download sentiment and spaCy models (~500 MB)
3. Create a default admin API key (check logs)

Models are cached in the `speechscore-models` Docker volume and persist across restarts.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SPEECHSCORE_PORT` | 8000 | API port |
| `SPEECHSCORE_MAX_QUEUE` | 20 | Max async queue depth |
| `SPEECHSCORE_JOB_EXPIRY_HOURS` | 24 | Job result retention |

### GPU Selection

```bash
# Use specific GPU
NVIDIA_VISIBLE_DEVICES=0 docker compose up -d

# In docker-compose.yml, change count: 1 to specific device
```

---

## Option B: Bare Metal (systemd)

### Prerequisites
- Python 3.12+
- NVIDIA GPU with CUDA 12.x
- ffmpeg

### Setup

```bash
# Clone and install
cd /home/jsuppe/speech3
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Create directories
mkdir -p api/logs api/tmp_uploads

# Test run
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Install as systemd service

```bash
# Copy service file
sudo cp deploy/speechscore.service /etc/systemd/system/

# Edit paths if needed
sudo systemctl edit speechscore

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable speechscore
sudo systemctl start speechscore

# Check status
sudo systemctl status speechscore
journalctl -u speechscore -f
```

### nginx Reverse Proxy (TLS)

```bash
# Install nginx + certbot
sudo apt install nginx certbot python3-certbot-nginx

# Get TLS cert
sudo certbot certonly --nginx -d api.speechscore.dev

# Install config
sudo cp deploy/nginx.conf /etc/nginx/sites-available/speechscore
sudo ln -s /etc/nginx/sites-available/speechscore /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

---

## Option C: Cloud GPU (Serverless)

### RunPod

1. Create a serverless endpoint with the Docker image
2. Select A10G or A100 GPU
3. Set min workers = 0 (scale to zero)
4. Configure health check: `GET /v1/health`
5. Timeout: 180s (for long audio files)

### Modal

```python
import modal

app = modal.App("speechscore")
image = modal.Image.from_dockerfile("Dockerfile")

@app.function(gpu="A10G", image=image, timeout=300)
@modal.web_endpoint(method="POST")
def analyze(audio: bytes):
    # Forward to internal API
    ...
```

---

## Production Checklist

### Security
- [ ] Change default admin API key
- [ ] Enable TLS (nginx + certbot or cloud load balancer)
- [ ] Set `allow_origins` in CORS middleware (remove `*`)
- [ ] Restrict admin endpoints to internal network
- [ ] Set up firewall rules (only expose 443)

### Monitoring
- [ ] Health check monitoring (uptime robot, etc.)
- [ ] GPU utilization alerts (nvidia-smi)
- [ ] Disk space monitoring (model cache + logs)
- [ ] Rate limit / usage dashboards

### Backup
- [ ] Back up `keys.json` (API keys)
- [ ] Back up `usage.json` (usage data)
- [ ] Log rotation (already configured in docker-compose)

### Performance
- [ ] Warm up models on deploy (health check confirms readiness)
- [ ] Set appropriate `max_queue_depth` for your GPU
- [ ] Monitor VRAM usage (Whisper large-v3 ~3 GB + overhead)
- [ ] Consider multiple workers for CPU-bound analysis modules

### Scaling
- Single GPU: ~100 three-min analyses/hour
- For higher throughput: run multiple containers on multiple GPUs
- Or use serverless (auto-scales with demand)

---

## Troubleshooting

### "CUDA out of memory"
- Whisper large-v3 needs ~3 GB VRAM
- If other processes use the GPU, try `CUDA_VISIBLE_DEVICES=0`
- For <6 GB VRAM, use `medium` model (edit `config.py`)

### Slow startup
- First run downloads ~3.5 GB of models
- Subsequent starts: ~15-20s (loading from cache)
- Set `start_period: 120s` in health check to allow warmup

### "ffmpeg not found"
- Install: `apt install ffmpeg` or `brew install ffmpeg`
- Docker image includes ffmpeg automatically

### GPU not detected
- Check: `nvidia-smi` works
- Docker: ensure `--gpus all` or nvidia runtime
- Check: `NVIDIA_VISIBLE_DEVICES` environment variable
