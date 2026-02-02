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
