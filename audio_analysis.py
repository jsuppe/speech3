"""
Audio-level speech analysis for the Melchior Voice Pipeline.
Analyzes pitch, cadence, pauses, volume dynamics, speaking rate,
and voice quality from raw audio using Praat (parselmouth).
"""
import numpy as np
import subprocess
import parselmouth
from parselmouth.praat import call


def load_sound(audio_path):
    """Load audio into Praat Sound object, converting if needed."""
    # Convert to WAV first if not already
    if not audio_path.endswith(".wav"):
        import tempfile, os
        wav_path = tempfile.mktemp(suffix=".wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", wav_path],
            capture_output=True, timeout=30
        )
        snd = parselmouth.Sound(wav_path)
        os.remove(wav_path)
    else:
        snd = parselmouth.Sound(audio_path)
    return snd


def analyze_pitch(snd, floor=75, ceiling=500):
    """
    Analyze fundamental frequency (F0) — the core of voice pitch.
    Returns stats on pitch: mean, median, std, min, max, range.
    """
    pitch = call(snd, "To Pitch", 0.0, floor, ceiling)

    # Extract voiced frames only
    pitch_values = pitch.selected_array['frequency']
    voiced = pitch_values[pitch_values > 0]

    if len(voiced) == 0:
        return {
            "mean_hz": 0, "median_hz": 0, "std_hz": 0,
            "min_hz": 0, "max_hz": 0, "range_hz": 0,
            "voiced_fraction": 0.0,
            "pitch_contour": []
        }

    # Pitch contour (sampled at pitch frame rate for visualization)
    times = pitch.xs()
    contour = []
    for t, f in zip(times, pitch_values):
        if f > 0:
            contour.append({"time": round(float(t), 3), "hz": round(float(f), 1)})

    return {
        "mean_hz": round(float(np.mean(voiced)), 1),
        "median_hz": round(float(np.median(voiced)), 1),
        "std_hz": round(float(np.std(voiced)), 1),
        "min_hz": round(float(np.min(voiced)), 1),
        "max_hz": round(float(np.max(voiced)), 1),
        "range_hz": round(float(np.max(voiced) - np.min(voiced)), 1),
        "voiced_fraction": round(float(len(voiced) / len(pitch_values)), 3),
    }


def analyze_intensity(snd):
    """
    Analyze volume/intensity dynamics.
    Returns mean, std, min, max, dynamic range in dB.
    """
    intensity = call(snd, "To Intensity", 75, 0.0, "yes")

    values = intensity.values[0]
    valid = values[values > 0]

    if len(valid) == 0:
        return {
            "mean_db": 0, "std_db": 0, "min_db": 0,
            "max_db": 0, "dynamic_range_db": 0
        }

    return {
        "mean_db": round(float(np.mean(valid)), 1),
        "std_db": round(float(np.std(valid)), 1),
        "min_db": round(float(np.min(valid)), 1),
        "max_db": round(float(np.max(valid)), 1),
        "dynamic_range_db": round(float(np.max(valid) - np.min(valid)), 1),
    }


def analyze_voice_quality(snd, floor=75, ceiling=500):
    """
    Analyze voice quality metrics: jitter (pitch stability) and shimmer (amplitude stability).
    These are clinical measures used in speech pathology.
    - Jitter: cycle-to-cycle pitch variation (higher = less stable voice)
    - Shimmer: cycle-to-cycle amplitude variation (higher = breathier/rougher voice)
    """
    try:
        point_process = call(snd, "To PointProcess (periodic, cc)", floor, ceiling)

        # Jitter (local) — typical healthy: < 1.04%
        jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)

        # Shimmer (local) — typical healthy: < 3.81%
        shimmer_local = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        # Harmonics-to-noise ratio (HNR) — higher = clearer voice, typical: > 20 dB
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, floor, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)

        return {
            "jitter_percent": round(float(jitter_local * 100), 3),
            "jitter_status": "normal" if jitter_local * 100 < 1.04 else "elevated",
            "shimmer_percent": round(float(shimmer_local * 100), 3),
            "shimmer_status": "normal" if shimmer_local * 100 < 3.81 else "elevated",
            "harmonics_to_noise_db": round(float(hnr), 1),
            "hnr_status": "clear" if hnr > 20 else ("moderate" if hnr > 10 else "rough"),
        }
    except Exception as e:
        return {
            "jitter_percent": None,
            "shimmer_percent": None,
            "harmonics_to_noise_db": None,
            "error": str(e)
        }


def analyze_pauses(segments, total_duration):
    """
    Analyze pause patterns from Whisper segment timestamps.
    segments: list of {"start": float, "end": float, "text": str}
    """
    if not segments or len(segments) < 2:
        speech_dur = segments[0]["end"] - segments[0]["start"] if segments else 0
        return {
            "pause_count": 0,
            "total_pause_sec": round(float(total_duration - speech_dur), 2),
            "mean_pause_sec": 0,
            "max_pause_sec": 0,
            "speech_to_silence_ratio": 0,
            "pauses": []
        }

    pauses = []
    for i in range(len(segments) - 1):
        gap = segments[i + 1]["start"] - segments[i]["end"]
        if gap > 0.15:  # Only count pauses > 150ms
            pauses.append({
                "start": round(float(segments[i]["end"]), 2),
                "end": round(float(segments[i + 1]["start"]), 2),
                "duration": round(float(gap), 2)
            })

    total_speech = sum(s["end"] - s["start"] for s in segments)
    total_pause = sum(p["duration"] for p in pauses)

    return {
        "pause_count": len(pauses),
        "total_pause_sec": round(float(total_pause), 2),
        "mean_pause_sec": round(float(np.mean([p["duration"] for p in pauses])), 2) if pauses else 0,
        "max_pause_sec": round(float(max(p["duration"] for p in pauses)), 2) if pauses else 0,
        "speech_to_silence_ratio": round(float(total_speech / max(total_pause, 0.01)), 2),
        "pauses": pauses
    }


def analyze_speaking_rate(segments, total_duration):
    """
    Calculate speaking rate metrics.
    - Overall WPM (including pauses)
    - Articulation rate (speech-only WPM, excluding pauses)
    """
    if not segments:
        return {
            "overall_wpm": 0,
            "articulation_wpm": 0,
            "total_words": 0,
            "speech_duration_sec": 0,
        }

    total_words = sum(len(s["text"].split()) for s in segments)
    speech_duration = sum(s["end"] - s["start"] for s in segments)

    overall_wpm = (total_words / total_duration) * 60 if total_duration > 0 else 0
    articulation_wpm = (total_words / speech_duration) * 60 if speech_duration > 0 else 0

    return {
        "overall_wpm": round(float(overall_wpm), 1),
        "articulation_wpm": round(float(articulation_wpm), 1),
        "total_words": total_words,
        "speech_duration_sec": round(float(speech_duration), 2),
    }


def analyze_audio(audio_path, segments, total_duration):
    """
    Full audio-level speech analysis.
    Returns dict with all audio metrics.
    """
    snd = load_sound(audio_path)

    pitch = analyze_pitch(snd)
    intensity = analyze_intensity(snd)
    voice_quality = analyze_voice_quality(snd)
    pauses = analyze_pauses(segments, total_duration)
    rate = analyze_speaking_rate(segments, total_duration)

    # Summary interpretation
    interpretations = []

    # Speaking rate
    wpm = rate["overall_wpm"]
    if wpm < 100:
        interpretations.append(f"Speaking rate is slow ({wpm:.0f} WPM) — average conversational is 120-150 WPM")
    elif wpm > 180:
        interpretations.append(f"Speaking rate is fast ({wpm:.0f} WPM) — may affect clarity")
    else:
        interpretations.append(f"Speaking rate is normal ({wpm:.0f} WPM)")

    # Pitch variation
    if pitch["std_hz"] > 0:
        cv = pitch["std_hz"] / pitch["mean_hz"] if pitch["mean_hz"] > 0 else 0
        if cv < 0.1:
            interpretations.append("Pitch variation is low — speech may sound monotone")
        elif cv > 0.3:
            interpretations.append("Pitch variation is high — expressive/animated delivery")
        else:
            interpretations.append("Pitch variation is normal — natural intonation")

    # Pauses
    if pauses["pause_count"] > 0 and pauses["mean_pause_sec"] > 1.5:
        interpretations.append(f"Long average pauses ({pauses['mean_pause_sec']:.1f}s) — may indicate hesitation or deliberate pacing")

    # Voice quality
    if voice_quality.get("hnr_status") == "rough":
        interpretations.append("Voice quality appears rough — could be environmental noise or vocal strain")

    return {
        "pitch": pitch,
        "intensity": intensity,
        "voice_quality": voice_quality,
        "pauses": pauses,
        "speaking_rate": rate,
        "interpretations": interpretations,
    }
