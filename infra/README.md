# SpeakFit Infrastructure

## Quick Start

### Prerequisites
1. Install [Terraform](https://terraform.io) >= 1.0
2. Install [Google Cloud CLI](https://cloud.google.com/sdk)
3. Create a GCP project and enable billing

### Setup

```bash
# Authenticate with GCP
gcloud auth application-default login

# Create project (if new)
gcloud projects create speakfit-prod --name="SpeakFit Production"
gcloud config set project speakfit-prod

# Enable required APIs
gcloud services enable \
  compute.googleapis.com \
  sqladmin.googleapis.com \
  redis.googleapis.com \
  run.googleapis.com \
  containerregistry.googleapis.com \
  servicenetworking.googleapis.com

# Initialize Terraform
cd terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values

terraform init
terraform plan
terraform apply
```

## Cost Estimates

### Development (~$150/mo)
- 1 API server (Cloud Run): $30
- PostgreSQL (db-custom-2-8192): $50
- Redis (1GB Basic): $25
- 1 GPU worker (T4 preemptible): $45
- Storage + networking: ~$10

### Production (~$800/mo)
- 2-3 API servers: $60-90
- PostgreSQL (db-custom-4-16384): $150
- Redis (4GB HA): $100
- 2 GPU workers (T4): $500
- Storage + networking: ~$50

### Scale (10K+ DAU, ~$2,500/mo)
- 5 API servers: $150
- PostgreSQL (read replicas): $400
- Redis Cluster: $200
- 5 GPU workers: $1,250
- Storage + CDN: $500

## Architecture Decisions

### Why GCP over AWS?
- Better GPU availability and pricing
- Cloud Run is simpler than ECS/Fargate
- Cloud SQL UI is cleaner than RDS
- Competitive pricing overall

### Why Cloud Run for API?
- Auto-scales to zero (dev cost savings)
- Built-in load balancing
- Simple deploys (just push container)
- No Kubernetes complexity

### Why Compute Engine for GPU?
- Cloud Run doesn't support GPUs
- More control over GPU instance type
- Can use preemptible for 60% savings
- Custom startup scripts for Whisper

### Why not serverless GPU (Modal/Replicate)?
- Great for low/bursty traffic
- We have consistent load → dedicated is cheaper
- Can switch later if traffic patterns change

## Migration from Current Setup

### Phase 1: Database (Week 1)
1. Provision Cloud SQL
2. Migrate SQLite → Postgres
3. Update connection strings
4. Test thoroughly

### Phase 2: Queue System (Week 2)
1. Add Redis
2. Refactor GPU work to Celery tasks
3. Test locally with Redis

### Phase 3: Cloud Deploy (Week 3)
1. Build Docker images
2. Deploy API to Cloud Run
3. Deploy GPU worker to Compute Engine
4. Switch DNS

### Phase 4: Production Hardening (Week 4)
1. Set up monitoring (Cloud Monitoring + PagerDuty)
2. Configure alerts
3. Load testing
4. Document runbooks

## Docker Images

### API Server
```dockerfile
# infra/docker/api/Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### GPU Worker
```dockerfile
# infra/docker/gpu-worker/Dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04
RUN pip install faster-whisper celery redis
COPY . .
CMD ["celery", "-A", "worker", "worker", "--loglevel=info"]
```

## Monitoring

### Key Metrics
- API latency (p50, p95, p99)
- GPU worker queue depth
- Transcription throughput (chunks/min)
- Error rates by endpoint
- Database connections

### Alerts
- API latency > 2s (warning), > 5s (critical)
- Queue depth > 100 (warning), > 500 (critical)
- Error rate > 1% (warning), > 5% (critical)
- GPU worker down for > 5 min (critical)

## Secrets Management

Sensitive values in Terraform:
- Database password (auto-generated)
- API keys (add via Secret Manager)

```bash
# Add ElevenLabs key
echo -n "your-api-key" | gcloud secrets create elevenlabs-api-key --data-file=-

# Add OAuth secrets
gcloud secrets create google-oauth-client-id --data-file=client_id.txt
```

## Rollback

```bash
# Revert to previous API version
gcloud run services update-traffic speakfit-prod-api \
  --to-revisions=speakfit-prod-api-00005-abc=100

# Terraform rollback
terraform plan -target=google_cloud_run_v2_service.api
terraform apply -target=google_cloud_run_v2_service.api
```
