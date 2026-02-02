# SpeechScore Python SDK

Python client for the [SpeechScore API](https://docs.speechscore.dev) — comprehensive speech analysis as a service.

## Installation

```bash
pip install speechscore
```

## Quick Start

```python
from speechscore import SpeechScore

client = SpeechScore(api_key="sk-YOUR-API-KEY")

# Analyze a file
result = client.analyze("speech.mp3")
print(result.transcript)
print(f"Words per minute: {result.wpm}")
print(f"Sentiment: {result.overall_sentiment}")
print(f"Vocal fry: {result.vocal_fry_pct}%")
print(f"Readability: grade {result.readability_grade}")
```

## Usage

### Sync Analysis

```python
# Full analysis (all modules)
result = client.analyze("speech.mp3")

# Quick transcription only
result = client.analyze("speech.mp3", preset="quick")
print(result.transcript)

# Select specific modules
result = client.analyze("speech.mp3", modules="transcription,audio,sentiment")

# Use presets: quick, text_only, audio_only, full, clinical, coaching
result = client.analyze("speech.mp3", preset="coaching")

# Set quality gate
result = client.analyze("speech.mp3", quality_gate="block")  # reject noisy audio

# Specify language (skip auto-detection)
result = client.analyze("speech.mp3", language="en")
```

### Async Analysis

```python
# Submit for background processing
job = client.analyze_async("long_speech.mp3")
print(f"Job submitted: {job.job_id}")

# Poll until complete
completed = client.wait_for_job(job.job_id, poll_interval=2.0)
print(completed.result.transcript)

# Or poll manually
job = client.get_job(job.job_id)
if job.is_complete:
    print(job.result.wpm)

# With webhook notification
job = client.analyze_async(
    "speech.mp3",
    webhook_url="https://example.com/webhook"
)
```

### Working with Results

```python
result = client.analyze("speech.mp3")

# Quick summary
print(result.summary())
# {'wpm': 142.5, 'mean_pitch_hz': 145.2, 'sentiment': 'POSITIVE', ...}

# Transcript and segments
print(result.transcript)
for seg in result.segments:
    print(f"[{seg['start']:.1f}s] {seg['text']}")

# Audio metrics
print(f"Pitch: {result.mean_pitch_hz} Hz")
print(f"HNR: {result.hnr_db} dB")
print(f"Vocal fry: {result.vocal_fry_pct}%")
print(f"WPM: {result.wpm}")

# Text metrics
print(f"Lexical diversity: {result.text.lexical_diversity}")
print(f"Readability: grade {result.readability_grade}")

# Rhetorical
if result.rhetorical.repetition:
    print(f"Repetition score: {result.rhetorical.repetition.repetition_score}")

# Sentiment
print(f"Sentiment: {result.overall_sentiment}")
print(f"Emotional arc: {result.sentiment.emotional_arc}")

# Audio quality
print(f"SNR: {result.snr_db} dB ({result.confidence})")

# Raw JSON access
print(result.raw)
```

### Queue & Usage

```python
# Check queue status
queue = client.queue_status()
print(f"Queue depth: {queue['queue_depth']}/{queue['max_queue_depth']}")

# Check your usage
usage = client.usage()
print(f"Requests today: {usage['current_period']['requests_today']}")
print(f"Audio minutes: {usage['all_time']['total_audio_minutes']}")

# Health check
health = client.health()
print(f"GPU: {health['gpu_name']}")
```

### File-like Objects

```python
# Works with file objects too
with open("speech.mp3", "rb") as f:
    result = client.analyze(f, filename="speech.mp3")

# Or from bytes
import io
audio_bytes = download_audio()
result = client.analyze(io.BytesIO(audio_bytes), filename="recording.wav")
```

### Error Handling

```python
from speechscore import SpeechScore, AuthError, RateLimitError, ValidationError

client = SpeechScore(api_key="sk-YOUR-KEY")

try:
    result = client.analyze("speech.mp3")
except AuthError:
    print("Invalid API key")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after}s")
except ValidationError as e:
    print(f"Bad request: {e.code} — {e.message}")
```

### Environment Variables

```bash
export SPEECHSCORE_API_KEY=sk-YOUR-KEY
export SPEECHSCORE_BASE_URL=https://api.speechscore.dev
```

```python
# No need to pass api_key or base_url
client = SpeechScore()
```

### Context Manager

```python
with SpeechScore(api_key="sk-YOUR-KEY") as client:
    result = client.analyze("speech.mp3")
```

## API Reference

See [full API documentation](https://docs.speechscore.dev).

## License

MIT
