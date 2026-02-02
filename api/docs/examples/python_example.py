"""SpeechScore API — Python examples using the SDK."""

from speechscore import SpeechScore, ValidationError, RateLimitError

client = SpeechScore(api_key="sk-YOUR-KEY")

# ── Basic Analysis ───────────────────────────────────────────────
result = client.analyze("speech.mp3")
print(f"Transcript: {result.transcript[:100]}...")
print(f"WPM: {result.wpm}")
print(f"Pitch: {result.mean_pitch_hz} Hz")
print(f"Sentiment: {result.overall_sentiment}")
print()

# ── Summary View ─────────────────────────────────────────────────
print("Summary:", result.summary())
print()

# ── Specific Modules Only ────────────────────────────────────────
quick = client.analyze("speech.mp3", preset="quick")
print(f"Quick transcript: {quick.transcript[:80]}...")
print(f"SNR: {quick.snr_db} dB ({quick.confidence})")
print()

# ── Async + Poll ─────────────────────────────────────────────────
job = client.analyze_async("long_lecture.mp3")
print(f"Submitted job: {job.job_id} (status: {job.status})")

# Wait for completion (polls every 2s, up to 5 min)
completed = client.wait_for_job(job.job_id, poll_interval=2.0, timeout=300)
print(f"Completed! WPM: {completed.result.wpm}")
print()

# ── Batch Analysis ───────────────────────────────────────────────
import glob

files = glob.glob("recordings/*.mp3")
for f in files:
    try:
        r = client.analyze(f, preset="coaching")
        print(f"{f}: {r.wpm:.0f} WPM, {r.overall_sentiment}, fry={r.vocal_fry_pct}%")
    except ValidationError as e:
        print(f"{f}: Skipped — {e.message}")
    except RateLimitError as e:
        print(f"Rate limited! Wait {e.retry_after}s")
        break

# ── Compare Speakers ─────────────────────────────────────────────
speakers = {
    "Alice": "alice_speech.mp3",
    "Bob": "bob_speech.mp3",
}

for name, path in speakers.items():
    r = client.analyze(path)
    print(f"\n{name}:")
    print(f"  WPM: {r.wpm:.0f}")
    print(f"  Pitch: {r.mean_pitch_hz:.0f} Hz")
    print(f"  Vocal fry: {r.vocal_fry_pct}%")
    print(f"  HNR: {r.hnr_db:.1f} dB")
    print(f"  Readability: grade {r.readability_grade}")
    print(f"  Sentiment: {r.overall_sentiment}")

# ── Clinical Screening ───────────────────────────────────────────
clinical = client.analyze("patient_sample.wav", preset="clinical")
vq = clinical.audio.voice_quality
print(f"\nClinical Voice Screen:")
print(f"  Jitter: {vq.jitter_percent}%")
print(f"  Shimmer: {vq.shimmer_percent}%")
print(f"  HNR: {vq.harmonics_to_noise_db} dB")
print(f"  Vocal fry: {clinical.vocal_fry_pct}%")
if clinical.advanced_audio.formants:
    print(f"  Vowel space: {clinical.advanced_audio.formants.vowel_space_area}")
