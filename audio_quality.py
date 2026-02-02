"""
Audio quality assessment for the Melchior Voice Pipeline.
Estimates SNR, background noise level, and provides a confidence rating.
"""
import numpy as np
import subprocess
import os
import json


def load_audio_as_pcm(audio_path, sample_rate=16000):
    """Load audio file as mono PCM float32 using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", audio_path,
        "-ar", str(sample_rate), "-ac", "1", "-f", "f32le", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0:
        return None, sample_rate
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    return audio, sample_rate


def estimate_snr(audio, sample_rate=16000, frame_ms=30):
    """
    Estimate SNR by comparing energy of speech frames vs noise frames.
    Uses a simple energy-based VAD to separate speech from silence/noise.
    """
    if audio is None or len(audio) == 0:
        return 0.0

    frame_size = int(sample_rate * frame_ms / 1000)
    num_frames = len(audio) // frame_size

    if num_frames == 0:
        return 0.0

    # Compute energy per frame
    energies = []
    for i in range(num_frames):
        frame = audio[i * frame_size:(i + 1) * frame_size]
        energy = np.mean(frame ** 2)
        energies.append(energy)

    energies = np.array(energies)

    if len(energies) == 0 or np.max(energies) == 0:
        return 0.0

    # Use energy threshold to separate speech from noise
    # Sort energies, bottom 20% is noise floor estimate
    sorted_e = np.sort(energies)
    noise_frames = sorted_e[:max(1, len(sorted_e) // 5)]
    noise_energy = np.mean(noise_frames)

    # Top frames are speech
    speech_frames = sorted_e[len(sorted_e) // 2:]  # upper half
    speech_energy = np.mean(speech_frames)

    if noise_energy <= 0:
        noise_energy = 1e-10

    snr_db = 10 * np.log10(speech_energy / noise_energy)
    return round(float(snr_db), 1)


def assess_quality(audio, sample_rate=16000):
    """
    Full audio quality assessment.
    Returns dict with SNR, noise level, clipping, and overall confidence.
    """
    if audio is None or len(audio) == 0:
        return {
            "snr_db": 0,
            "noise_floor_db": -100,
            "peak_amplitude": 0,
            "clipping_ratio": 0,
            "duration_sec": 0,
            "confidence": "UNUSABLE",
            "confidence_score": 0.0,
            "issues": ["No audio data"]
        }

    duration = len(audio) / sample_rate
    snr = estimate_snr(audio, sample_rate)

    # Noise floor (RMS of quietest 20% of frames)
    frame_size = int(sample_rate * 0.03)
    num_frames = len(audio) // frame_size
    energies = []
    for i in range(num_frames):
        frame = audio[i * frame_size:(i + 1) * frame_size]
        energies.append(np.sqrt(np.mean(frame ** 2)))
    
    sorted_rms = np.sort(energies) if energies else [0]
    noise_rms = np.mean(sorted_rms[:max(1, len(sorted_rms) // 5)])
    noise_floor_db = round(float(20 * np.log10(max(noise_rms, 1e-10))), 1)

    # Peak and clipping
    peak = float(np.max(np.abs(audio)))
    clipping = float(np.mean(np.abs(audio) > 0.99))

    # Issues list
    issues = []
    if snr < 5:
        issues.append(f"Very low SNR ({snr} dB) — heavy background noise")
    elif snr < 10:
        issues.append(f"Low SNR ({snr} dB) — noticeable background noise")
    if clipping > 0.01:
        issues.append(f"Audio clipping detected ({clipping*100:.1f}% of samples)")
    if peak < 0.05:
        issues.append("Very quiet audio — may be too far from mic")
    if duration < 2:
        issues.append("Very short recording — limited analysis accuracy")
    if noise_floor_db > -20:
        issues.append(f"High noise floor ({noise_floor_db} dB)")

    # Confidence rating
    score = 1.0
    if snr < 5:
        score -= 0.5
    elif snr < 10:
        score -= 0.3
    elif snr < 15:
        score -= 0.1
    if clipping > 0.01:
        score -= 0.2
    if peak < 0.05:
        score -= 0.3
    if duration < 2:
        score -= 0.2
    score = max(0.0, min(1.0, score))

    if score >= 0.8:
        confidence = "HIGH"
    elif score >= 0.6:
        confidence = "MEDIUM"
    elif score >= 0.3:
        confidence = "LOW"
    else:
        confidence = "UNUSABLE"

    return {
        "snr_db": float(snr),
        "noise_floor_db": float(noise_floor_db),
        "peak_amplitude": round(float(peak), 4),
        "clipping_ratio": round(float(clipping), 4),
        "duration_sec": round(float(duration), 2),
        "confidence": confidence,
        "confidence_score": round(float(score), 2),
        "issues": issues
    }
