# SpeakFit Cloud Architecture

## Overview

```
                                    ┌─────────────────┐
                                    │   CloudFlare    │
                                    │   (CDN + WAF)   │
                                    └────────┬────────┘
                                             │
                                    ┌────────▼────────┐
                                    │  Load Balancer  │
                                    │   (GCP/AWS)     │
                                    └────────┬────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
           ┌────────▼────────┐     ┌────────▼────────┐     ┌────────▼────────┐
           │   API Server    │     │   API Server    │     │   API Server    │
           │   (FastAPI)     │     │   (FastAPI)     │     │   (FastAPI)     │
           │   CPU: 4 vCPU   │     │   CPU: 4 vCPU   │     │   CPU: 4 vCPU   │
           └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
                    │                        │                        │
                    └────────────────────────┼────────────────────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
           ┌────────▼────────┐     ┌────────▼────────┐     ┌────────▼────────┐
           │     Redis       │     │    Postgres     │     │   Object Store  │
           │  (Job Queue)    │     │   (Cloud SQL)   │     │   (GCS/S3)      │
           └─────────────────┘     └─────────────────┘     └─────────────────┘

                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
           ┌────────▼────────┐     ┌────────▼────────┐     ┌────────▼────────┐
           │   GPU Worker    │     │   GPU Worker    │     │   GPU Worker    │
           │   (Whisper)     │     │   (Whisper)     │     │   (Whisper)     │
           │   T4/L4 GPU     │     │   T4/L4 GPU     │     │   T4/L4 GPU     │
           └─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Components

### 1. CDN + WAF (CloudFlare)
- DDoS protection
- SSL termination
- Static asset caching
- Bot protection
- **Cost**: Free tier → $20/mo Pro

### 2. Load Balancer
- Health checks on API servers
- SSL passthrough or termination
- Auto-scaling triggers
- **Cost**: ~$20/mo (GCP) or ~$25/mo (AWS ALB)

### 3. API Servers (Stateless)
- FastAPI application
- Handles auth, routing, non-GPU tasks
- Praat/audio analysis (CPU-bound)
- Auto-scales 2-10 instances
- **Specs**: 4 vCPU, 8GB RAM, no GPU
- **Cost**: ~$50/mo each (e2-standard-4)

### 4. Redis (Managed)
- Job queue (Celery/RQ)
- Session cache
- Rate limiting
- **Cost**: ~$25/mo (Redis Cloud free tier) → $100/mo (production)

### 5. PostgreSQL (Managed)
- Cloud SQL (GCP) or RDS (AWS)
- 2 vCPU, 8GB RAM, 100GB SSD
- Automatic backups, replicas
- **Cost**: ~$50/mo (db-custom-2-8192)

### 6. Object Storage
- Audio file storage (recordings)
- TTS cache
- Signed URLs for secure access
- **Cost**: ~$0.02/GB/mo + egress

### 7. GPU Workers
- Whisper transcription
- Runs as separate service, pulls from Redis queue
- Can scale independently
- **Options**:
  - GCP T4: ~$0.35/hr ($250/mo)
  - GCP L4: ~$0.70/hr ($500/mo)
  - RunPod 3090: ~$0.40/hr ($290/mo)
  - Modal/Replicate: Pay per second

## Scaling Tiers

### Tier 1: 1,000 DAU (~$200/mo)
- 1 API server
- 1 GPU worker (or your 3090)
- Managed Postgres (small)
- Redis Cloud free tier

### Tier 2: 10,000 DAU (~$800/mo)
- 2-3 API servers (auto-scale)
- 2 GPU workers
- Managed Postgres (medium)
- Redis Cloud paid

### Tier 3: 100,000 DAU (~$5,000/mo)
- 5-10 API servers
- 5+ GPU workers (or switch to Deepgram API)
- Postgres with read replicas
- Redis Cluster

## Alternative: Serverless GPU

For variable load, consider serverless GPU:
- **Modal**: $0.000016/sec for T4 (~$0.06/hr when running)
- **Replicate**: Similar pricing
- **Beam**: Good for batch jobs

Pay only when processing, great for:
- Low volume / bursty traffic
- Testing / staging environments
- Cost optimization at scale

## Data Strategy

### Cloud Database (Postgres)
Only user data goes to the cloud:
- Users (5 → thousands)
- User recordings (~14MB currently)
- Analyses, transcriptions
- Coach messages, projects
- **Total: ~50MB now, scales with users**

### Local Database (SQLite)
Training/benchmark data stays local:
- 114K batch imports (LibriSpeech, LJSpeech)
- Reference speeches
- ML training datasets
- **Total: 15GB, doesn't need cloud**

### Migration Path

1. **Phase 1**: Install Postgres locally, migrate user data
2. **Phase 2**: Update API to use DATABASE_URL env var
3. **Phase 3**: Add Redis queue, decouple GPU work
4. **Phase 4**: Deploy to cloud (GCP/AWS)
5. **Phase 5**: Auto-scaling, monitoring, alerts
