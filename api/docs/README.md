# SpeechScore API

**Comprehensive speech analysis as a service.** Upload audio, get back transcription, text analysis, rhetorical analysis, sentiment, voice quality metrics, and more.

Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (large-v3) with a 10-module analysis pipeline powered by Praat, spaCy, and custom NLP.

## Quick Start

### 1. Get an API Key

Contact the administrator or use the admin endpoint to create a key.

### 2. Analyze Audio

```bash
curl -X POST https://api.speechscore.dev/v1/analyze \
  -H "Authorization: Bearer sk-YOUR-API-KEY" \
  -F "audio=@speech.mp3"
```

### 3. Get Results

The response includes transcription, text metrics, audio metrics, sentiment, rhetorical analysis, and more. See [API Reference](#api-reference) below.

---

## Installation (Python SDK)

```bash
pip install speechscore
```

```python
from speechscore import SpeechScore

client = SpeechScore(api_key="sk-YOUR-API-KEY")
result = client.analyze("speech.mp3")

print(result.transcript)
print(f"WPM: {result.audio.speaking_rate.overall_wpm}")
print(f"Sentiment: {result.sentiment.overall.label}")
```

---

## Authentication

All endpoints except `/v1/health` require an API key.

Pass it via:
- **Header (recommended):** `Authorization: Bearer sk-YOUR-API-KEY`
- **Query param:** `?api_key=sk-YOUR-API-KEY`

### Rate Limits

| Tier | Requests/min | Audio min/day |
|------|-------------|---------------|
| Free | 5 | 10 |
| Pro | 30 | 500 |
| Enterprise | 100 | Unlimited |

Rate limit headers are included on every response:
- `X-RateLimit-Limit` — max requests per minute
- `X-RateLimit-Remaining` — remaining requests this window
- `X-RateLimit-Reset` — seconds until window resets

---

## API Reference

### `GET /v1/health`

Health check. No authentication required.

**Response:**
```json
{
  "status": "ok",
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3090",
  "whisper_model_loaded": true,
  "whisper_model": "large-v3",
  "uptime_sec": 3600.0
}
```

---

### `POST /v1/analyze`

Analyze an uploaded audio file. Supports sync (default) and async modes.

**Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `audio` | file | *required* | Audio file (mp3, wav, aac, ogg, m4a, flac, webm) |
| `mode` | query | `sync` | Set to `async` for background processing |
| `modules` | query | `all` | Comma-separated: `transcription,text,audio,advanced_text,rhetorical,sentiment,advanced_audio,quality` |
| `preset` | query | — | Named preset (overrides `modules`): `quick`, `text_only`, `audio_only`, `full`, `clinical`, `coaching` |
| `quality_gate` | query | `warn` | `block` (reject bad audio), `warn` (default), `skip` (fastest) |
| `language` | query | auto | Language code for Whisper (e.g. `en`, `es`, `fr`) |
| `webhook_url` | query | — | URL to POST results when async job completes |

**Limits:**
- Max file size: 50 MB
- Max audio duration: 10 minutes
- Accepted formats: mp3, wav, aac, ogg, m4a, flac, webm

#### Sync Response (200 OK)

```json
{
  "audio_file": "speech.mp3",
  "audio_duration_sec": 180.0,
  "language": "en",
  "language_probability": 0.99,
  "transcript": "Full transcription text...",
  "segments": [
    {"start": 0.0, "end": 4.5, "text": "First sentence."},
    {"start": 4.5, "end": 9.2, "text": "Second sentence."}
  ],
  "audio_quality": {
    "snr_db": 25.3,
    "confidence": "HIGH",
    "confidence_score": 0.92,
    "issues": []
  },
  "speech3_text_analysis": {
    "metrics": {
      "lexical_diversity": 0.72,
      "syntactic_complexity": 1.45,
      "fluency": 0.88,
      "connectedness": 0.65,
      "mlu": 12.3
    },
    "key_terms": ["economy", "education", "future"]
  },
  "advanced_text_analysis": {
    "readability": {
      "flesch_reading_ease": 65.2,
      "flesch_kincaid_grade": 8.4,
      "automated_readability_index": 9.1
    },
    "discourse_structure": { "sentences": 15, "paragraphs": 3 },
    "questions": { "count": 2, "density": 0.13 },
    "named_entities": [
      {"text": "America", "label": "GPE", "count": 3}
    ],
    "perspective": {
      "first_person": 0.15,
      "second_person": 0.08,
      "third_person": 0.22,
      "inclusive_we": 0.12
    }
  },
  "rhetorical_analysis": {
    "repetition": {
      "anaphora": [{"phrase": "We must", "count": 4}],
      "epistrophe": [],
      "repeated_phrases": [{"phrase": "our nation", "count": 3}],
      "repetition_score": 5.35,
      "rhetorical_repetition": true,
      "sentence_count": 15
    },
    "rhythm": {
      "pause_regularity": 0.72,
      "mean_pause_sec": 0.45
    }
  },
  "sentiment": {
    "overall": {"label": "POSITIVE", "score": 0.82},
    "per_segment": [
      {"start": 0.0, "end": 4.5, "label": "NEUTRAL", "score": 0.55}
    ],
    "emotional_arc": "ascending"
  },
  "audio_analysis": {
    "pitch": {
      "mean_hz": 145.2,
      "min_hz": 85.0,
      "max_hz": 320.0,
      "std_hz": 42.5,
      "range_hz": 235.0
    },
    "intensity": {
      "mean_db": 72.3,
      "std_db": 8.1
    },
    "voice_quality": {
      "jitter_percent": 1.2,
      "shimmer_percent": 8.5,
      "harmonics_to_noise_db": 11.2
    },
    "pauses": {
      "count": 12,
      "total_sec": 15.3,
      "mean_sec": 1.28,
      "longest_sec": 3.2
    },
    "speaking_rate": {
      "overall_wpm": 142.5,
      "articulation_wpm": 165.0,
      "total_words": 425,
      "speech_duration_sec": 178.9
    }
  },
  "advanced_audio_analysis": {
    "formants": {
      "f1_mean": 520.3,
      "f2_mean": 1650.8,
      "f3_mean": 2800.1,
      "dispersion_hz": 1140.2,
      "vowel_space_area": 285000.0
    },
    "pitch_contour": {
      "slope": -0.12,
      "trend": "slightly_falling",
      "uptalk_ratio": 0.05,
      "uptalk_detected": false
    },
    "rate_variability": {
      "coefficient_of_variation": 0.25,
      "segment_rates": [140.2, 155.0, 128.5]
    },
    "vocal_fry": {
      "vocal_fry_ratio": 0.02,
      "vocal_fry_detected": false,
      "fry_description": "minimal/no vocal fry"
    },
    "emphasis_patterns": {
      "emphasis_count": 8,
      "emphasis_rate": 0.044
    }
  },
  "processing": {
    "duration_s": 180.0,
    "processing_time_s": 32.5,
    "modules_requested": ["all"],
    "modules_run": ["transcription", "quality", "text", "advanced_text", "rhetorical", "sentiment", "audio", "advanced_audio"],
    "preset": null,
    "quality_gate": "warn",
    "timing": {
      "transcription_sec": 8.2,
      "text_analysis_sec": 0.3,
      "audio_analysis_sec": 12.1,
      "total_sec": 32.5
    }
  }
}
```

#### Async Response (202 Accepted)

When `mode=async`:

```json
{
  "job_id": "c42e663d-011c-45f0-96e7-8e07cfd8f9c6",
  "status": "queued",
  "created_at": "2026-02-01T22:15:00.000Z"
}
```

---

### `GET /v1/jobs/{job_id}`

Poll the status of an async job.

**Response (processing):**
```json
{
  "job_id": "c42e663d-...",
  "status": "processing",
  "filename": "speech.mp3",
  "created_at": "2026-02-01T22:15:00.000Z",
  "started_at": "2026-02-01T22:15:01.000Z",
  "progress": "processing:advanced_audio_analysis"
}
```

**Response (complete):**
```json
{
  "job_id": "c42e663d-...",
  "status": "complete",
  "filename": "speech.mp3",
  "created_at": "2026-02-01T22:15:00.000Z",
  "started_at": "2026-02-01T22:15:01.000Z",
  "completed_at": "2026-02-01T22:15:33.000Z",
  "result": { ... }
}
```

**Statuses:** `queued` → `processing` → `complete` | `failed`

---

### `GET /v1/jobs`

List recent jobs (last 50).

| Param | Type | Description |
|-------|------|-------------|
| `status` | query | Filter: `queued`, `processing`, `complete`, `failed` |

---

### `GET /v1/queue`

Queue status and estimated wait time.

```json
{
  "queue_depth": 2,
  "max_queue_depth": 20,
  "currently_processing": "c42e663d-...",
  "currently_processing_filename": "speech.mp3",
  "estimated_wait_sec": 65.0,
  "jobs_queued": 1,
  "jobs_processing": 1,
  "jobs_complete": 42,
  "jobs_failed": 0
}
```

---

### `GET /v1/usage`

Usage stats for the current API key.

```json
{
  "key_id": "49b06996",
  "key_name": "my-app",
  "tier": "pro",
  "current_period": {
    "requests_today": 42,
    "audio_minutes_today": 15.3,
    "requests_limit_per_minute": 30,
    "audio_minutes_limit_per_day": 500
  },
  "all_time": {
    "total_requests": 1250,
    "total_audio_minutes": 450.7
  }
}
```

---

## Module Reference

| Module | Key | What It Measures |
|--------|-----|-----------------|
| Transcription | `transcription` | Whisper STT with timestamps |
| Audio Quality | `quality` | SNR, noise floor, clipping, confidence |
| Text Analysis | `text` | Lexical diversity, syntactic complexity, fluency, MLU |
| Advanced Text | `advanced_text` | Readability (Flesch), discourse, entities, perspective |
| Rhetorical | `rhetorical` | Anaphora, repetition score, rhythm |
| Sentiment | `sentiment` | 3-class (pos/neg/neutral), emotional arc |
| Audio | `audio` | Pitch (F0), intensity, jitter/shimmer/HNR, pauses, WPM |
| Advanced Audio | `advanced_audio` | Formants, vocal fry, pitch contour, uptalk, emphasis |

### Presets

| Preset | Modules | Typical Speed |
|--------|---------|---------------|
| `quick` | transcription, quality | ~0.8s |
| `text_only` | transcription, quality, text, advanced_text, rhetorical, sentiment | ~1.5s |
| `audio_only` | transcription, quality, audio, advanced_audio | ~2s |
| `clinical` | transcription, quality, audio, advanced_audio | ~2s |
| `coaching` | transcription, quality, text, audio, sentiment | ~2s |
| `full` | everything | ~3-30s |

---

## Error Codes

| Code | HTTP | Description |
|------|------|-------------|
| `MISSING_FILE` | 400 | No audio file in request |
| `INVALID_FILE_TYPE` | 400 | Unsupported file extension |
| `EMPTY_FILE` | 400 | File has zero bytes |
| `FILE_TOO_LARGE` | 413 | Exceeds 50MB limit |
| `CORRUPT_AUDIO` | 422 | Cannot decode audio |
| `AUDIO_TOO_LONG` | 422 | Exceeds 10-minute limit |
| `QUALITY_GATE_BLOCKED` | 422 | Audio too noisy (when `quality_gate=block`) |
| `INVALID_PRESET` | 400 | Unknown preset name |
| `INVALID_MODULES` | 400 | Unknown module name(s) |
| `QUEUE_FULL` | 429 | Async queue at capacity (20 jobs) |
| `RATE_LIMITED` | 429 | Rate limit exceeded |
| `UNAUTHORIZED` | 401 | Missing or invalid API key |
| `FORBIDDEN` | 403 | Key disabled or insufficient tier |
| `JOB_NOT_FOUND` | 404 | Job ID not found or expired (24h) |
| `PIPELINE_ERROR` | 500 | Internal analysis error |

All errors follow the format:
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable description."
  }
}
```

---

## Webhook Notifications

For async jobs, pass `webhook_url` to receive results via POST when complete:

```bash
curl -X POST https://api.speechscore.dev/v1/analyze?mode=async&webhook_url=https://example.com/hook \
  -H "Authorization: Bearer sk-YOUR-KEY" \
  -F "audio=@speech.mp3"
```

The webhook payload is identical to the job status response with `result` populated.

---

## Interactive Docs

When the server is running, visit:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

---

## Self-Hosting

```bash
cd /path/to/speech3
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
bash api/run.sh
```

**Requirements:** Python 3.12+, NVIDIA GPU with CUDA, ffmpeg

See [DEPLOYMENT.md](DEPLOYMENT.md) for Docker and production setup.
