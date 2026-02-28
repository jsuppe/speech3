"""
Sentiment analysis for the Melchior Voice Pipeline.
Uses a transformer model for nuanced sentiment on transcript text.
Analyzes both overall sentiment and per-segment sentiment arc.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import pipeline as hf_pipeline

# Lazy-load the model
_sentiment_pipe = None

def _get_pipe():
    global _sentiment_pipe
    if _sentiment_pipe is None:
        _sentiment_pipe = hf_pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device="cpu",
            top_k=None
        )
    return _sentiment_pipe


def analyze_sentiment(transcript, segments=None):
    """
    Analyze sentiment of transcript text.
    
    Returns:
        - overall: sentiment of the full transcript
        - per_segment: sentiment arc across segments (if provided)
        - emotional_arc: simplified trajectory (positive/negative/neutral over time)
    """
    if not transcript or not transcript.strip():
        return {
            "overall": {"label": "NEUTRAL", "score": 0.0},
            "per_segment": [],
            "emotional_arc": "flat"
        }

    pipe = _get_pipe()

    # Overall sentiment
    overall_result = pipe(transcript[:512])[0]  # truncate to model max
    overall = _parse_scores(overall_result)

    # Per-segment sentiment
    per_segment = []
    if segments:
        for seg in segments:
            text = seg.get("text", "").strip()
            if len(text) < 3:
                continue
            seg_result = pipe(text)[0]
            parsed = _parse_scores(seg_result)
            per_segment.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": text[:80],
                "sentiment": parsed["label"],
                "positive_score": parsed["positive"],
                "negative_score": parsed["negative"],
            })

    # Emotional arc summary
    arc = _summarize_arc(per_segment) if per_segment else "flat"

    return {
        "overall": overall,
        "per_segment": per_segment,
        "emotional_arc": arc,
    }


def _parse_scores(result_list):
    """Parse huggingface sentiment output into normalized format."""
    scores = {r["label"].lower(): r["score"] for r in result_list}
    pos = scores.get("positive", 0)
    neg = scores.get("negative", 0)
    neu = scores.get("neutral", 0)

    top = max(scores, key=scores.get)
    label = top.upper()

    return {
        "label": label,
        "positive": round(float(pos), 4),
        "negative": round(float(neg), 4),
        "neutral": round(float(neu), 4),
        "compound": round(float(pos - neg), 4),
    }


def _summarize_arc(per_segment):
    """Summarize the emotional trajectory across segments."""
    if not per_segment:
        return "flat"

    compounds = [s["positive_score"] - s["negative_score"] for s in per_segment]

    if len(compounds) < 3:
        avg = sum(compounds) / len(compounds)
        if avg > 0.3:
            return "positive"
        elif avg < -0.3:
            return "negative"
        return "neutral"

    # Split into first half / second half
    mid = len(compounds) // 2
    first_avg = sum(compounds[:mid]) / mid
    second_avg = sum(compounds[mid:]) / (len(compounds) - mid)

    diff = second_avg - first_avg

    if abs(diff) < 0.15:
        if (first_avg + second_avg) / 2 > 0.3:
            return "consistently positive"
        elif (first_avg + second_avg) / 2 < -0.3:
            return "consistently negative"
        return "neutral/stable"
    elif diff > 0.15:
        return "rising (becomes more positive)"
    else:
        return "falling (becomes more negative)"


def compute_normalized_arc(per_segment):
    """
    Compute a normalized emotional arc for real-time playback visualization.
    
    Returns arc data normalized to the speech's own emotional range,
    with phase detection (Setup, Build, Peak, Resolution).
    
    Returns:
        - points: list of {time_pct, intensity, phase, raw_sentiment}
        - peak_position: where the emotional peak occurs (0-100%)
        - arc_type: classification of the arc shape
        - phases: list of {name, start_pct, end_pct}
    """
    if not per_segment or len(per_segment) < 2:
        return {
            "points": [],
            "peak_position": 50,
            "arc_type": "flat",
            "phases": [],
            "intensity_range": {"min": 0, "max": 0},
        }
    
    # Extract compound scores (positive - negative) as raw intensity
    compounds = [s["positive_score"] - s["negative_score"] for s in per_segment]
    
    # Get timing info
    total_duration = per_segment[-1]["end"]
    
    # Normalize to 0-100 intensity scale relative to THIS speech's range
    min_val = min(compounds)
    max_val = max(compounds)
    val_range = max_val - min_val if max_val != min_val else 1
    
    # Build normalized points
    points = []
    for i, seg in enumerate(per_segment):
        time_pct = (seg["start"] / total_duration) * 100 if total_duration > 0 else 0
        raw = compounds[i]
        # Normalize: map [min_val, max_val] -> [0, 100]
        intensity = ((raw - min_val) / val_range) * 100
        
        points.append({
            "time_pct": round(time_pct, 1),
            "time_sec": round(seg["start"], 2),
            "intensity": round(intensity, 1),
            "raw_sentiment": round(raw, 3),
            "text_preview": seg.get("text", "")[:40],
        })
    
    # Find peak position
    max_intensity_idx = compounds.index(max(compounds))
    peak_position = points[max_intensity_idx]["time_pct"] if points else 50
    
    # Detect phases based on intensity patterns
    phases = _detect_phases(points, peak_position)
    
    # Classify arc type
    arc_type = _classify_arc_type(points, peak_position)
    
    # Assign phase to each point
    for point in points:
        point["phase"] = _get_phase_at_position(point["time_pct"], phases)
    
    return {
        "points": points,
        "peak_position": round(peak_position, 1),
        "arc_type": arc_type,
        "phases": phases,
        "intensity_range": {
            "min": round(min_val, 3),
            "max": round(max_val, 3),
        },
        "total_duration_sec": round(total_duration, 2),
    }


def _detect_phases(points, peak_position):
    """Detect the four oratory phases: Setup, Build, Peak, Resolution."""
    if not points:
        return []
    
    # Default phase boundaries based on peak position
    # Ideal oratory: Setup (0-25%), Build (25-peak), Peak (around peak), Resolution (peak-100%)
    
    if peak_position < 30:
        # Early peak - unusual, adjust phases
        phases = [
            {"name": "Opening", "start_pct": 0, "end_pct": peak_position - 5, "emoji": "📉"},
            {"name": "Peak", "start_pct": peak_position - 5, "end_pct": peak_position + 10, "emoji": "🔥"},
            {"name": "Development", "start_pct": peak_position + 10, "end_pct": 75, "emoji": "📊"},
            {"name": "Resolution", "start_pct": 75, "end_pct": 100, "emoji": "📈"},
        ]
    elif peak_position > 80:
        # Late peak - building speech
        phases = [
            {"name": "Setup", "start_pct": 0, "end_pct": 30, "emoji": "📉"},
            {"name": "Build", "start_pct": 30, "end_pct": peak_position - 5, "emoji": "📊"},
            {"name": "Climax", "start_pct": peak_position - 5, "end_pct": 100, "emoji": "🔥"},
        ]
    else:
        # Classic arc with peak in middle-to-late
        phases = [
            {"name": "Setup", "start_pct": 0, "end_pct": 25, "emoji": "📉"},
            {"name": "Build", "start_pct": 25, "end_pct": peak_position - 5, "emoji": "📊"},
            {"name": "Peak", "start_pct": peak_position - 5, "end_pct": peak_position + 10, "emoji": "🔥"},
            {"name": "Resolution", "start_pct": peak_position + 10, "end_pct": 100, "emoji": "📈"},
        ]
    
    return phases


def _classify_arc_type(points, peak_position):
    """Classify the overall arc shape."""
    if not points or len(points) < 3:
        return "flat"
    
    intensities = [p["intensity"] for p in points]
    
    # Check variance
    avg = sum(intensities) / len(intensities)
    variance = sum((x - avg) ** 2 for x in intensities) / len(intensities)
    
    if variance < 100:  # Low variance = flat
        return "steady"
    
    # Check if it builds to peak
    first_quarter = intensities[:len(intensities)//4] if len(intensities) >= 4 else intensities[:1]
    last_quarter = intensities[-len(intensities)//4:] if len(intensities) >= 4 else intensities[-1:]
    
    first_avg = sum(first_quarter) / len(first_quarter)
    last_avg = sum(last_quarter) / len(last_quarter)
    
    if 50 <= peak_position <= 80 and first_avg < 40 and max(intensities) > 70:
        return "classic_build"  # Like MLK
    elif peak_position < 30:
        return "front_loaded"
    elif peak_position > 80:
        return "crescendo"
    elif last_avg > first_avg + 20:
        return "rising"
    elif first_avg > last_avg + 20:
        return "falling"
    else:
        return "dynamic"


def _get_phase_at_position(time_pct, phases):
    """Get the phase name for a given time position."""
    for phase in phases:
        if phase["start_pct"] <= time_pct < phase["end_pct"]:
            return phase["name"]
    return phases[-1]["name"] if phases else "Unknown"
