"""
Quick confidence/nervousness analysis for Live Coach.
Analyzes short audio clips (~1.5s) to detect nervous speaking patterns.
"""

import tempfile
import subprocess
import os
import numpy as np

# Lazy load parselmouth (heavy import)
_parselmouth = None
_call = None

def _load_parselmouth():
    global _parselmouth, _call
    if _parselmouth is None:
        import parselmouth
        from parselmouth.praat import call
        _parselmouth = parselmouth
        _call = call
    return _parselmouth, _call


def analyze_confidence_quick(audio_path: str, text: str = "") -> dict:
    """
    Quick confidence analysis optimized for ~1.5-3 second audio clips.
    
    Returns:
        - confidence_score: 0-100 (0=very nervous, 100=very confident)
        - confidence_level: "nervous", "slightly_nervous", "neutral", "confident"
        - indicators: dict of individual metrics
    """
    parselmouth, call = _load_parselmouth()
    
    indicators = {}
    
    try:
        # Convert to WAV if needed
        if not audio_path.endswith(".wav"):
            wav_path = tempfile.mktemp(suffix=".wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", wav_path],
                capture_output=True, timeout=10
            )
            snd = parselmouth.Sound(wav_path)
            os.unlink(wav_path)
        else:
            snd = parselmouth.Sound(audio_path)
        
        # Get duration
        duration = snd.get_total_duration()
        if duration < 0.5:
            return {
                "confidence_score": 50,
                "confidence_level": "neutral",
                "indicators": {"error": "audio too short"}
            }
        
        # 1. Pitch analysis (nervous = higher pitch, more variation)
        pitch = call(snd, "To Pitch", 0.0, 75, 500)
        pitch_values = pitch.selected_array['frequency']
        voiced = pitch_values[pitch_values > 0]
        
        if len(voiced) > 5:
            pitch_mean = float(np.mean(voiced))
            pitch_std = float(np.std(voiced))
            pitch_cv = pitch_std / pitch_mean if pitch_mean > 0 else 0
            
            indicators["pitch_mean_hz"] = round(pitch_mean, 1)
            indicators["pitch_std_hz"] = round(pitch_std, 1)
            indicators["pitch_cv"] = round(pitch_cv, 3)
            
            # High pitch deviation from baseline suggests stress
            # CV > 0.25 suggests erratic pitch (nervous), < 0.10 is monotone (different issue)
            pitch_stability_score = 100 - min(100, abs(pitch_cv - 0.15) * 400)
        else:
            pitch_stability_score = 50
            indicators["pitch_mean_hz"] = 0
        
        # 2. Jitter (voice tremor - higher = more nervous)
        try:
            point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
            jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_pct = jitter * 100
            
            indicators["jitter_percent"] = round(jitter_pct, 3)
            
            # Normal jitter < 1%, stressed voice often > 2%
            if jitter_pct < 1.0:
                jitter_score = 100
            elif jitter_pct < 2.0:
                jitter_score = 80 - (jitter_pct - 1.0) * 40
            else:
                jitter_score = max(0, 40 - (jitter_pct - 2.0) * 20)
        except:
            jitter_score = 50
            indicators["jitter_percent"] = None
        
        # 3. Intensity variation (confident = more dynamic, nervous = flat or erratic)
        try:
            intensity = call(snd, "To Intensity", 75, 0.0, "yes")
            int_values = intensity.values[0]
            int_values = int_values[int_values > 0]
            
            if len(int_values) > 5:
                int_std = float(np.std(int_values))
                indicators["intensity_std_db"] = round(int_std, 1)
                
                # Good range is 5-15 dB variation
                if 5 <= int_std <= 15:
                    intensity_score = 100
                else:
                    intensity_score = max(0, 100 - abs(int_std - 10) * 8)
            else:
                intensity_score = 50
        except:
            intensity_score = 50
        
        # 4. Text-based hedging (nervous speakers hedge more)
        hedging_score = 100
        if text:
            text_lower = text.lower()
            hedging_phrases = [
                "i think", "maybe", "sort of", "kind of", "i guess",
                "probably", "perhaps", "i'm not sure", "i don't know",
                "basically", "actually", "just", "like"
            ]
            hedging_count = sum(1 for p in hedging_phrases if p in text_lower)
            word_count = len(text.split())
            
            if word_count > 0:
                hedge_ratio = hedging_count / word_count
                indicators["hedging_ratio"] = round(hedge_ratio, 3)
                # More than 15% hedging words = nervous
                hedging_score = max(0, 100 - hedge_ratio * 500)
            
        # 5. Speaking rate from text (if available)
        wpm_score = 100
        if text and duration > 0:
            word_count = len(text.split())
            wpm = (word_count / duration) * 60
            indicators["wpm"] = round(wpm, 1)
            
            # Ideal: 130-160 WPM. Nervous rushing: >180. Nervous slow: <100
            if 130 <= wpm <= 160:
                wpm_score = 100
            elif 100 <= wpm < 130 or 160 < wpm <= 180:
                wpm_score = 70
            else:
                wpm_score = max(0, 50 - abs(wpm - 145) * 0.5)
        
        # Combine scores with weights
        # Jitter and pitch stability are strongest voice-based indicators
        weights = {
            "pitch_stability": 0.25,
            "jitter": 0.30,
            "intensity": 0.10,
            "hedging": 0.20,
            "wpm": 0.15
        }
        
        confidence_score = (
            pitch_stability_score * weights["pitch_stability"] +
            jitter_score * weights["jitter"] +
            intensity_score * weights["intensity"] +
            hedging_score * weights["hedging"] +
            wpm_score * weights["wpm"]
        )
        
        confidence_score = max(0, min(100, round(confidence_score)))
        
        # Map to levels
        if confidence_score >= 75:
            level = "confident"
        elif confidence_score >= 55:
            level = "neutral"
        elif confidence_score >= 35:
            level = "slightly_nervous"
        else:
            level = "nervous"
        
        indicators["component_scores"] = {
            "pitch_stability": round(pitch_stability_score),
            "jitter": round(jitter_score),
            "intensity": round(intensity_score),
            "hedging": round(hedging_score),
            "wpm": round(wpm_score)
        }
        
        return {
            "confidence_score": confidence_score,
            "confidence_level": level,
            "indicators": indicators
        }
        
    except Exception as e:
        return {
            "confidence_score": 50,
            "confidence_level": "neutral",
            "indicators": {"error": str(e)}
        }
