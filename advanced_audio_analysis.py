"""
Advanced audio analysis for the Melchior Voice Pipeline.
Covers formants, pitch contour trends, speaking rate variability,
vocal fry detection, and emphasis patterns.
"""
import numpy as np
import subprocess
import parselmouth
from parselmouth.praat import call


def load_sound(audio_path):
    """Load audio into Praat Sound object, converting if needed."""
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


def analyze_formants(snd, max_formant=5500):
    """
    Analyze vowel formants (F1, F2, F3).
    - F1 correlates with jaw openness (higher = more open)
    - F2 correlates with tongue position (higher = more front)
    - F1-F2 vowel space area indicates articulation precision
    - Larger vowel space = clearer pronunciation
    """
    try:
        formants = call(snd, "To Formant (burg)", 0.0, 5, max_formant, 0.025, 50.0)
        
        f1_values = []
        f2_values = []
        f3_values = []
        
        num_frames = call(formants, "Get number of frames")
        for i in range(1, num_frames + 1):
            f1 = call(formants, "Get value at time", 1, call(formants, "Get time from frame number", i), "hertz", "Linear")
            f2 = call(formants, "Get value at time", 2, call(formants, "Get time from frame number", i), "hertz", "Linear")
            f3 = call(formants, "Get value at time", 3, call(formants, "Get time from frame number", i), "hertz", "Linear")
            
            if not np.isnan(f1) and f1 > 0:
                f1_values.append(f1)
            if not np.isnan(f2) and f2 > 0:
                f2_values.append(f2)
            if not np.isnan(f3) and f3 > 0:
                f3_values.append(f3)
        
        if not f1_values or not f2_values:
            return {"error": "Could not extract formants"}
        
        # Vowel space estimate: std of F1 × std of F2 
        # Larger = more distinct vowel articulation
        f1_std = float(np.std(f1_values))
        f2_std = float(np.std(f2_values))
        vowel_space = f1_std * f2_std
        
        return {
            "f1_mean_hz": round(float(np.mean(f1_values)), 1),
            "f1_std_hz": round(f1_std, 1),
            "f2_mean_hz": round(float(np.mean(f2_values)), 1),
            "f2_std_hz": round(f2_std, 1),
            "f3_mean_hz": round(float(np.mean(f3_values)), 1) if f3_values else 0,
            "vowel_space_index": round(vowel_space / 1000, 2),  # Normalized
            "articulation_clarity": "clear" if vowel_space > 80000 else ("moderate" if vowel_space > 40000 else "compressed"),
        }
    except Exception as e:
        return {"error": str(e)}


def analyze_pitch_contour(snd, segments, floor=75, ceiling=500):
    """
    Analyze pitch contour trends:
    - Rising vs falling intonation at utterance ends
    - Uptalk detection (rising intonation on statements)
    - Pitch reset at topic boundaries
    - Overall pitch trajectory across the speech
    """
    try:
        pitch = call(snd, "To Pitch", 0.0, floor, ceiling)
        
        # Per-segment pitch trends
        segment_trends = []
        rising_count = 0
        falling_count = 0
        
        for seg in segments:
            start = seg["start"]
            end = seg["end"]
            duration = end - start
            
            if duration < 0.5:
                continue
            
            # Sample pitch at last 30% of segment (ending intonation)
            end_start = start + duration * 0.7
            
            # Get pitch values in the ending portion
            times = pitch.xs()
            values = pitch.selected_array['frequency']
            
            end_pitches = []
            for t, f in zip(times, values):
                if end_start <= t <= end and f > 0:
                    end_pitches.append(f)
            
            if len(end_pitches) >= 3:
                # Linear trend of ending pitch
                x = np.arange(len(end_pitches))
                slope = np.polyfit(x, end_pitches, 1)[0]
                
                if slope > 2:
                    trend = "rising"
                    rising_count += 1
                elif slope < -2:
                    trend = "falling"
                    falling_count += 1
                else:
                    trend = "flat"
                
                segment_trends.append({
                    "start": round(start, 2),
                    "end_pitch_slope": round(float(slope), 2),
                    "trend": trend,
                })
        
        total_trends = rising_count + falling_count
        uptalk_ratio = rising_count / max(total_trends, 1)
        
        # Overall pitch trajectory (beginning vs end of speech)
        all_times = pitch.xs()
        all_values = pitch.selected_array['frequency']
        voiced = [(t, f) for t, f in zip(all_times, all_values) if f > 0]
        
        overall_trajectory = "stable"
        if len(voiced) > 20:
            quarter = len(voiced) // 4
            first_q_mean = np.mean([f for _, f in voiced[:quarter]])
            last_q_mean = np.mean([f for _, f in voiced[-quarter:]])
            diff = last_q_mean - first_q_mean
            if diff > 15:
                overall_trajectory = "rising (builds energy)"
            elif diff < -15:
                overall_trajectory = "falling (winding down)"
        
        return {
            "rising_endings": rising_count,
            "falling_endings": falling_count,
            "uptalk_ratio": round(float(uptalk_ratio), 3),
            "uptalk_detected": uptalk_ratio > 0.4,
            "overall_pitch_trajectory": overall_trajectory,
            "segment_trends": segment_trends[:20],  # Cap for readability
        }
    except Exception as e:
        return {"error": str(e)}


def analyze_rate_variability(segments):
    """
    Analyze speaking rate variability across segments.
    - Rate std: how much WPM varies (higher = more dynamic delivery)
    - Deceleration patterns: slowing for emphasis
    - Acceleration patterns: building energy
    """
    if not segments or len(segments) < 3:
        return {
            "rate_std_wpm": 0,
            "rate_cv": 0,
            "rate_dynamic": "insufficient data",
            "slowest_segment": None,
            "fastest_segment": None,
        }
    
    rates = []
    for s in segments:
        dur = s["end"] - s["start"]
        if dur > 0.3:
            words = len(s["text"].split())
            wpm = (words / dur) * 60
            rates.append({
                "start": s["start"],
                "wpm": wpm,
                "text": s["text"][:60]
            })
    
    if len(rates) < 3:
        return {"rate_std_wpm": 0, "rate_cv": 0, "rate_dynamic": "insufficient data"}
    
    wpm_values = [r["wpm"] for r in rates]
    rate_mean = float(np.mean(wpm_values))
    rate_std = float(np.std(wpm_values))
    rate_cv = rate_std / rate_mean if rate_mean > 0 else 0
    
    # Find slowest and fastest segments (likely emphasis/excitement)
    slowest = min(rates, key=lambda r: r["wpm"])
    fastest = max(rates, key=lambda r: r["wpm"])
    
    # Characterize dynamics
    if rate_cv < 0.15:
        dynamic = "monotonous (very steady rate)"
    elif rate_cv < 0.30:
        dynamic = "moderate variation (natural)"
    elif rate_cv < 0.50:
        dynamic = "dynamic (deliberate speed changes)"
    else:
        dynamic = "highly variable (dramatic pacing)"
    
    return {
        "rate_mean_wpm": round(rate_mean, 1),
        "rate_std_wpm": round(rate_std, 1),
        "rate_cv": round(float(rate_cv), 3),
        "rate_dynamic": dynamic,
        "slowest_segment": {
            "wpm": round(float(slowest["wpm"]), 1),
            "text": slowest["text"]
        },
        "fastest_segment": {
            "wpm": round(float(fastest["wpm"]), 1),
            "text": fastest["text"]
        },
    }


def detect_vocal_fry(snd, floor=40, ceiling=80):
    """
    Detect vocal fry (creaky voice) — very low frequency phonation.
    Common in casual modern speech, generally absent in trained speakers.
    Measured by looking for pitch values below typical speech range.
    """
    try:
        # Use a lower floor to catch vocal fry frequencies
        pitch = call(snd, "To Pitch", 0.0, floor, 500)
        values = pitch.selected_array['frequency']
        voiced = values[values > 0]
        
        if len(voiced) == 0:
            return {"vocal_fry_ratio": 0, "vocal_fry_detected": False}
        
        # Vocal fry typically falls below 70-80 Hz
        fry_frames = voiced[voiced < ceiling]
        fry_ratio = len(fry_frames) / len(voiced)
        
        return {
            "vocal_fry_ratio": round(float(fry_ratio), 3),
            "vocal_fry_detected": fry_ratio > 0.1,
            "fry_description": (
                "significant vocal fry detected" if fry_ratio > 0.2
                else "some vocal fry present" if fry_ratio > 0.1
                else "minimal/no vocal fry"
            ),
        }
    except Exception as e:
        return {"error": str(e)}


def analyze_emphasis_patterns(snd, segments):
    """
    Detect emphasis patterns — where the speaker puts extra energy.
    Combines pitch peaks and intensity peaks to find stressed segments.
    """
    if not segments or len(segments) < 3:
        return {"emphasis_segments": [], "emphasis_distribution": "insufficient data"}
    
    try:
        pitch = call(snd, "To Pitch", 0.0, 75, 500)
        intensity = call(snd, "To Intensity", 75, 0.0, "yes")
        
        segment_energy = []
        for seg in segments:
            mid = (seg["start"] + seg["end"]) / 2
            
            # Get mean pitch and intensity for this segment
            seg_pitch = call(pitch, "Get mean", seg["start"], seg["end"], "hertz")
            seg_intensity = call(intensity, "Get mean", seg["start"], seg["end"], "energy")
            
            if not np.isnan(seg_pitch) and not np.isnan(seg_intensity):
                segment_energy.append({
                    "start": round(seg["start"], 2),
                    "end": round(seg["end"], 2),
                    "text": seg["text"][:60],
                    "pitch": round(float(seg_pitch), 1),
                    "intensity": round(float(seg_intensity), 4),
                })
        
        if not segment_energy:
            return {"emphasis_segments": [], "emphasis_distribution": "no data"}
        
        # Normalize and find segments with above-average energy
        pitches = [s["pitch"] for s in segment_energy if s["pitch"] > 0]
        intensities = [s["intensity"] for s in segment_energy if s["intensity"] > 0]
        
        if not pitches or not intensities:
            return {"emphasis_segments": [], "emphasis_distribution": "no data"}
        
        pitch_mean = np.mean(pitches)
        pitch_std = np.std(pitches) if len(pitches) > 1 else 1
        int_mean = np.mean(intensities)
        int_std = np.std(intensities) if len(intensities) > 1 else 1
        
        emphasized = []
        for s in segment_energy:
            if s["pitch"] > 0 and s["intensity"] > 0:
                pitch_z = (s["pitch"] - pitch_mean) / max(pitch_std, 0.01)
                int_z = (s["intensity"] - int_mean) / max(int_std, 0.01)
                combined_z = (pitch_z + int_z) / 2
                
                if combined_z > 0.5:
                    emphasized.append({
                        "start": s["start"],
                        "text": s["text"],
                        "emphasis_score": round(float(combined_z), 2),
                    })
        
        # Distribution: front-loaded, back-loaded, or distributed?
        if emphasized:
            positions = [e["start"] for e in emphasized]
            total_dur = segments[-1]["end"]
            avg_pos = np.mean(positions) / total_dur if total_dur > 0 else 0.5
            
            if avg_pos < 0.35:
                distribution = "front-loaded (emphasis early)"
            elif avg_pos > 0.65:
                distribution = "back-loaded (builds to climax)"
            else:
                distribution = "distributed (emphasis throughout)"
        else:
            distribution = "flat (no clear emphasis peaks)"
        
        return {
            "emphasis_segments": sorted(emphasized, key=lambda x: -x["emphasis_score"])[:5],
            "emphasis_count": len(emphasized),
            "emphasis_distribution": distribution,
        }
    except Exception as e:
        return {"error": str(e)}


def analyze_advanced_audio(audio_path, segments):
    """Full advanced audio analysis."""
    snd = load_sound(audio_path)
    
    formants = analyze_formants(snd)
    pitch_contour = analyze_pitch_contour(snd, segments)
    rate_variability = analyze_rate_variability(segments)
    vocal_fry = detect_vocal_fry(snd)
    emphasis = analyze_emphasis_patterns(snd, segments)
    
    return {
        "formants": formants,
        "pitch_contour": pitch_contour,
        "rate_variability": rate_variability,
        "vocal_fry": vocal_fry,
        "emphasis_patterns": emphasis,
    }
