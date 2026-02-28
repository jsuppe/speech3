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

    # Calculate pitch variation as a 0-100 score
    # Based on coefficient of variation (std/mean) scaled to useful range
    cv = float(np.std(voiced) / np.mean(voiced)) if np.mean(voiced) > 0 else 0
    # CV of 0.15-0.25 is typical for expressive speech
    # Scale: 0 CV = 0 score, 0.3 CV = 100 score
    pitch_variation_score = min(100, int(cv / 0.003))
    
    return {
        "mean_hz": round(float(np.mean(voiced)), 1),
        "median_hz": round(float(np.median(voiced)), 1),
        "std_hz": round(float(np.std(voiced)), 1),
        "min_hz": round(float(np.min(voiced)), 1),
        "max_hz": round(float(np.max(voiced)), 1),
        "range_hz": round(float(np.max(voiced) - np.min(voiced)), 1),
        "voiced_fraction": round(float(len(voiced) / len(pitch_values)), 3),
        "pitch_contour": contour,  # For visualization
        "pitch_variation_score": pitch_variation_score,
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


def analyze_voice_health(snd, segments=None, target_duration=None, exercise_type=None):
    """
    Voice health specific analysis for voice health exercises.
    
    Returns metrics relevant to vocal health:
    - Sustained sound analysis (duration, steadiness, fade)
    - Vocal strain indicators
    - Exercise-specific feedback
    """
    results = {}
    
    duration = snd.get_total_duration()
    floor, ceiling = 75, 500
    
    # =========================================================================
    # 1. SUSTAINED SOUND ANALYSIS (for vowel sustain exercises)
    # =========================================================================
    try:
        pitch = call(snd, "To Pitch", 0.0, floor, ceiling)
        pitch_values = pitch.selected_array['frequency']
        voiced = pitch_values[pitch_values > 0]
        times = pitch.xs()
        
        # Duration of actual phonation (voiced sound)
        voiced_duration = len(voiced) * (times[1] - times[0]) if len(times) > 1 else 0
        
        # Pitch steadiness during sustained sound (lower std = steadier)
        pitch_steadiness = 0
        if len(voiced) > 10:
            # Use coefficient of variation (std/mean) - lower is better
            cv = float(np.std(voiced) / np.mean(voiced)) if np.mean(voiced) > 0 else 1.0
            # Convert to 0-100 score where 100 = very steady
            # CV of 0.02 = perfect, CV of 0.15 = poor
            pitch_steadiness = max(0, min(100, int((0.15 - cv) / 0.13 * 100)))
        
        results["sustained_sound"] = {
            "voiced_duration_sec": round(float(voiced_duration), 2),
            "target_duration_sec": target_duration,
            "duration_achieved_pct": round(float(voiced_duration / target_duration * 100), 1) if target_duration else None,
            "pitch_steadiness_score": pitch_steadiness,
            "pitch_steadiness_status": "excellent" if pitch_steadiness >= 80 else ("good" if pitch_steadiness >= 60 else ("fair" if pitch_steadiness >= 40 else "needs_work")),
        }
    except Exception as e:
        results["sustained_sound"] = {"error": str(e)}
    
    # =========================================================================
    # 2. FADE/BREATH SUPPORT ANALYSIS
    # =========================================================================
    try:
        intensity = call(snd, "To Intensity", 75, 0.0, "yes")
        int_values = intensity.values[0]
        int_times = intensity.xs()
        
        valid_mask = int_values > 0
        valid_values = int_values[valid_mask]
        
        if len(valid_values) > 20:
            # Compare first third vs last third intensity
            third = len(valid_values) // 3
            first_third_mean = np.mean(valid_values[:third])
            last_third_mean = np.mean(valid_values[-third:])
            
            # Fade amount in dB (negative = voice weakened)
            fade_db = float(last_third_mean - first_third_mean)
            
            # Breath support score: 100 = no fade, 0 = significant fade (>6dB drop)
            breath_support = max(0, min(100, int((6 + fade_db) / 6 * 100)))
            
            results["breath_support"] = {
                "fade_db": round(fade_db, 1),
                "breath_support_score": breath_support,
                "breath_support_status": "excellent" if breath_support >= 80 else ("good" if breath_support >= 60 else ("fair" if breath_support >= 40 else "weak")),
                "interpretation": "Strong breath support" if fade_db > -2 else ("Mild fade at end" if fade_db > -4 else "Voice weakened significantly - focus on diaphragmatic breathing")
            }
        else:
            results["breath_support"] = {"error": "Not enough audio data"}
    except Exception as e:
        results["breath_support"] = {"error": str(e)}
    
    # =========================================================================
    # 3. VOCAL STRAIN DETECTION
    # =========================================================================
    try:
        # Formant analysis for tension detection
        formants = call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
        
        # Get F1 and F2 values
        f1_values = []
        f2_values = []
        for t in np.linspace(0.1, duration - 0.1, 20):
            try:
                f1 = call(formants, "Get value at time", 1, t, "Hertz", "Linear")
                f2 = call(formants, "Get value at time", 2, t, "Hertz", "Linear")
                if f1 > 0 and f2 > 0:
                    f1_values.append(f1)
                    f2_values.append(f2)
            except:
                pass
        
        tension_indicators = []
        tension_score = 100  # Start at 100, deduct for issues
        
        if f1_values:
            f1_mean = np.mean(f1_values)
            f2_mean = np.mean(f2_values)
            
            # Elevated F1 (>800 Hz for sustained vowels) can indicate tension
            if f1_mean > 800:
                tension_indicators.append("Elevated F1 suggests possible throat tension")
                tension_score -= 20
            
            # F1/F2 ratio analysis
            ratio = f2_mean / f1_mean if f1_mean > 0 else 0
            if ratio < 1.5:
                tension_indicators.append("Compressed formant space - try relaxing jaw")
                tension_score -= 15
        
        # Jitter elevation during sustained sound indicates strain
        point_process = call(snd, "To PointProcess (periodic, cc)", floor, ceiling)
        jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        
        if jitter * 100 > 1.5:
            tension_indicators.append("Elevated jitter suggests vocal fatigue or strain")
            tension_score -= 25
        elif jitter * 100 > 1.04:
            tension_indicators.append("Slightly elevated pitch instability")
            tension_score -= 10
        
        # Glottal attack analysis (onset characteristics)
        # Analyze the first 100ms of voiced sound
        try:
            start_idx = np.argmax(int_values > np.max(int_values) * 0.3)
            onset_window = int_values[start_idx:start_idx + 10] if start_idx + 10 < len(int_values) else int_values[start_idx:]
            
            if len(onset_window) > 3:
                # Sharp rise = hard attack, gradual = soft attack
                onset_slope = (onset_window[-1] - onset_window[0]) / len(onset_window) if len(onset_window) > 1 else 0
                
                # Hard attack detection (slope > 5 dB per frame)
                if onset_slope > 5:
                    glottal_attack = "hard"
                    tension_indicators.append("Hard glottal attack detected - try softer onset")
                    tension_score -= 15
                elif onset_slope > 3:
                    glottal_attack = "moderate"
                else:
                    glottal_attack = "soft"
            else:
                glottal_attack = "unknown"
        except:
            glottal_attack = "unknown"
        
        tension_score = max(0, tension_score)
        
        results["strain_analysis"] = {
            "tension_score": tension_score,
            "tension_status": "relaxed" if tension_score >= 80 else ("mild_tension" if tension_score >= 60 else ("moderate_tension" if tension_score >= 40 else "significant_tension")),
            "glottal_attack": glottal_attack,
            "tension_indicators": tension_indicators if tension_indicators else ["No significant tension detected"],
            "f1_mean_hz": round(float(np.mean(f1_values)), 1) if f1_values else None,
            "f2_mean_hz": round(float(np.mean(f2_values)), 1) if f2_values else None,
        }
    except Exception as e:
        results["strain_analysis"] = {"error": str(e)}
    
    # =========================================================================
    # 4. EXERCISE-SPECIFIC ANALYSIS
    # =========================================================================
    if exercise_type:
        try:
            if exercise_type in ["humming", "humming_scales", "warmup_humming"]:
                # Analyze pitch glide smoothness for humming exercises
                if len(voiced) > 20:
                    # Calculate pitch differences between consecutive frames
                    pitch_diffs = np.abs(np.diff(voiced))
                    # Large jumps (>20 Hz) indicate choppy glide
                    smooth_frames = np.sum(pitch_diffs < 20)
                    total_frames = len(pitch_diffs)
                    smoothness = int(smooth_frames / total_frames * 100) if total_frames > 0 else 0
                    
                    results["exercise_feedback"] = {
                        "exercise_type": exercise_type,
                        "glide_smoothness_score": smoothness,
                        "glide_status": "smooth" if smoothness >= 80 else ("mostly_smooth" if smoothness >= 60 else "choppy"),
                        "pitch_range_used_hz": round(float(np.max(voiced) - np.min(voiced)), 1),
                        "tip": "Great smooth glides!" if smoothness >= 80 else "Try to connect notes more smoothly without jumps"
                    }
                    
            elif exercise_type in ["projection", "voice_projection"]:
                # Analyze projection: volume without strain
                intensity = call(snd, "To Intensity", 75, 0.0, "yes")
                mean_db = float(np.mean(intensity.values[0][intensity.values[0] > 0]))
                
                # Good projection = high volume + low jitter
                jitter_ok = results.get("strain_analysis", {}).get("tension_score", 50) >= 60
                loud_enough = mean_db > 70
                
                projection_score = 50
                if loud_enough:
                    projection_score += 25
                if jitter_ok:
                    projection_score += 25
                
                results["exercise_feedback"] = {
                    "exercise_type": exercise_type,
                    "mean_volume_db": round(mean_db, 1),
                    "projection_score": projection_score,
                    "projection_status": "excellent" if projection_score >= 90 else ("good" if projection_score >= 70 else "needs_work"),
                    "tip": "Great projection with good support!" if projection_score >= 90 else ("Louder but watch for strain" if not jitter_ok else "Try projecting more from your diaphragm")
                }
                
            elif exercise_type in ["resonance", "voice_resonance"]:
                # Forward resonance analysis via formant placement
                formants = call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
                
                f1_vals = []
                f2_vals = []
                f3_vals = []
                for t in np.linspace(0.1, duration - 0.1, 15):
                    try:
                        f1_vals.append(call(formants, "Get value at time", 1, t, "Hertz", "Linear"))
                        f2_vals.append(call(formants, "Get value at time", 2, t, "Hertz", "Linear"))
                        f3_vals.append(call(formants, "Get value at time", 3, t, "Hertz", "Linear"))
                    except:
                        pass
                
                f1_vals = [v for v in f1_vals if v > 0]
                f2_vals = [v for v in f2_vals if v > 0]
                f3_vals = [v for v in f3_vals if v > 0]
                
                # Forward resonance indicated by higher F2/F1 ratio and prominent F3
                resonance_score = 50
                tip = ""
                
                if f1_vals and f2_vals:
                    f2f1_ratio = np.mean(f2_vals) / np.mean(f1_vals)
                    if f2f1_ratio > 2.0:
                        resonance_score += 25
                        tip = "Good forward placement! "
                    else:
                        tip = "Try bringing the sound more forward in your mouth. "
                
                if f3_vals and np.mean(f3_vals) > 2500:
                    resonance_score += 25
                    tip += "Nice brightness in your tone!"
                else:
                    tip += "Focus on the 'mmm' vibration in your mask/face area."
                
                results["exercise_feedback"] = {
                    "exercise_type": exercise_type,
                    "resonance_score": resonance_score,
                    "resonance_status": "excellent" if resonance_score >= 90 else ("good" if resonance_score >= 70 else "developing"),
                    "f2_f1_ratio": round(np.mean(f2_vals) / np.mean(f1_vals), 2) if f1_vals and f2_vals else None,
                    "tip": tip.strip()
                }
                
            elif exercise_type in ["sustained_ah", "voice_sustained_ah", "vowel_sustain"]:
                # Already covered in sustained_sound section
                results["exercise_feedback"] = {
                    "exercise_type": exercise_type,
                    "tip": "Focus on keeping the pitch and volume steady throughout" if results.get("sustained_sound", {}).get("pitch_steadiness_score", 0) < 70 else "Excellent sustained tone!"
                }
        except Exception as e:
            results["exercise_feedback"] = {"exercise_type": exercise_type, "error": str(e)}
    
    # =========================================================================
    # 5. OVERALL VOICE HEALTH SCORE
    # =========================================================================
    try:
        scores = []
        
        if "sustained_sound" in results and "pitch_steadiness_score" in results["sustained_sound"]:
            scores.append(results["sustained_sound"]["pitch_steadiness_score"])
        
        if "breath_support" in results and "breath_support_score" in results["breath_support"]:
            scores.append(results["breath_support"]["breath_support_score"])
        
        if "strain_analysis" in results and "tension_score" in results["strain_analysis"]:
            scores.append(results["strain_analysis"]["tension_score"])
        
        overall_score = int(np.mean(scores)) if scores else None
        
        results["overall"] = {
            "voice_health_score": overall_score,
            "voice_health_status": "excellent" if overall_score and overall_score >= 80 else ("good" if overall_score and overall_score >= 60 else ("fair" if overall_score and overall_score >= 40 else "needs_attention")),
            "component_scores": {
                "pitch_steadiness": results.get("sustained_sound", {}).get("pitch_steadiness_score"),
                "breath_support": results.get("breath_support", {}).get("breath_support_score"),
                "relaxation": results.get("strain_analysis", {}).get("tension_score"),
            }
        }
    except Exception as e:
        results["overall"] = {"error": str(e)}
    
    return results


def analyze_audio(audio_path, segments, total_duration, voice_health_mode=False, target_duration=None, exercise_type=None):
    """
    Full audio-level speech analysis.
    Returns dict with all audio metrics.
    
    If voice_health_mode=True, includes detailed voice health analysis.
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

    result = {
        "pitch": pitch,
        "intensity": intensity,
        "voice_quality": voice_quality,
        "pauses": pauses,
        "speaking_rate": rate,
        "interpretations": interpretations,
    }
    
    # Add voice health analysis if requested
    if voice_health_mode:
        result["voice_health"] = analyze_voice_health(
            snd, 
            segments=segments, 
            target_duration=target_duration,
            exercise_type=exercise_type
        )
    
    return result
