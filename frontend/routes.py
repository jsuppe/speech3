"""Frontend page routes — serves Jinja2 templates."""

import os
import sys
import json
import logging
from pathlib import Path

from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

SPEECH3_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SPEECH3_DIR not in sys.path:
    sys.path.insert(0, SPEECH3_DIR)

from speech_db import SpeechDB
from scoring import score_speech, benchmark_speech, SCORING_PROFILES

logger = logging.getLogger("speechscore.frontend")

# Human-readable metric explanations used by tooltips across all templates
METRIC_EXPLANATIONS = {
    # Key metrics grid
    "WPM": "Words Per Minute — how fast the speaker talks. Deliberate orators: 100–130. Conversational: 140–170. Fast readers/debaters: 180+.",
    "Pitch": "Average fundamental frequency (F0) of the voice in Hertz. Typical male: 85–180 Hz. Typical female: 165–255 Hz. Higher pitch often signals energy or stress.",
    "Jitter": "Cycle-to-cycle variation in vocal pitch. Low (<1%) = steady, healthy voice. High (>2%) may indicate vocal strain, aging, or neurological conditions.",
    "Shimmer": "Cycle-to-cycle variation in vocal loudness. Low (<5%) = smooth voice. High values suggest breathiness, hoarseness, or reduced vocal control.",
    "HNR": "Harmonics-to-Noise Ratio — how clear and resonant the voice sounds. Higher = cleaner. 20+ dB is excellent; below 7 dB suggests significant hoarseness.",
    "Vocal Fry": "Ratio of creaky, low-frequency vocal pulses. Common in casual modern speech (20–50%). Rare in trained speakers (<5%). Not inherently bad — it's a style marker.",
    "Lex Diversity": "Lexical Diversity — ratio of unique words to total words (type-token ratio). Higher = richer vocabulary. Gehrig's farewell: 0.84. Typical speech: 0.40–0.65.",
    "Complexity": "Syntactic Complexity — average parse-tree depth of sentences. Higher = more embedded clauses and complex grammar. Academic: 15+. Casual: 8–12.",
    "Grade Level": "Flesch-Kincaid Grade Level — the U.S. school grade needed to understand the text. Grade 6 = easy for most. Grade 12+ = college-level complexity.",
    "Repetition": "Rhetorical repetition score — measures deliberate reuse of phrases for emphasis. MLK's 'I Have a Dream': 24.5. Most speakers: 0–3. High scores signal oratorical craft.",
    "Sentiment": "Overall emotional tone of the speech — positive, negative, or neutral, determined by a RoBERTa language model trained on social media text.",
    "Quality": "Audio signal quality level (HIGH / MEDIUM / LOW / UNUSABLE) based on Signal-to-Noise Ratio. HIGH means clean recording suitable for reliable analysis.",

    # Scored metrics (may overlap with above)
    "pitch_std_hz": "Pitch Standard Deviation — how much the voice pitch varies. High variation = expressive, dynamic delivery. Low = monotone. Great speakers: 30–60 Hz std.",
    "pitch_range_hz": "The span between lowest and highest pitch used. Wider range = more vocal expression. Narrow range can sound flat or robotic.",
    "uptalk_ratio": "Fraction of phrases ending with rising pitch (like questions). Common in uncertain speech. Low in authoritative delivery.",
    "rate_variability_cv": "Coefficient of variation in speaking rate — how much speed changes between segments. Some variation is natural; too much may indicate disfluency.",
    "fluency_score": "Overall fluency rating combining pause patterns, hesitations, and flow. Higher = smoother, more connected speech.",
    "flesch_reading_ease": "Flesch Reading Ease — how easy the text is to read. 90–100 = very easy (5th grade). 30–50 = college level. 0–30 = very difficult.",
    "anaphora_count": "Count of anaphoric repetitions — phrases repeated at the start of consecutive sentences/clauses. A classic rhetorical device (e.g., 'We shall fight…').",
    "rhythm_regularity": "How evenly spaced the stressed syllables are. Higher = more metered, speech-like cadence. Lower = choppy or irregular pacing.",
    "snr_db": "Signal-to-Noise Ratio in decibels — how much louder the speech is versus background noise. 30+ dB = studio quality. <10 dB = very noisy.",
    "lexical_diversity": "Lexical Diversity — ratio of unique words to total words (type-token ratio). Higher = richer vocabulary.",
    "syntactic_complexity": "Average parse-tree depth of sentences. Higher = more embedded clauses and complex grammar.",
    "fk_grade_level": "Flesch-Kincaid Grade Level — the U.S. school grade needed to understand the text.",
    "repetition_score": "Rhetorical repetition score — measures deliberate reuse of phrases for emphasis.",
    "wpm": "Words Per Minute — how fast the speaker talks.",
    "pitch_mean_hz": "Average fundamental frequency (F0) of the voice in Hertz.",
    "jitter_pct": "Cycle-to-cycle variation in vocal pitch percentage.",
    "shimmer_pct": "Cycle-to-cycle variation in vocal loudness percentage.",
    "hnr_db": "Harmonics-to-Noise Ratio — how clear and resonant the voice sounds.",
    "vocal_fry_ratio": "Ratio of creaky, low-frequency vocal pulses in the speech.",
    "quality_level": "Audio signal quality level based on Signal-to-Noise Ratio.",

    # Scoring system display labels (matched by m.label in metric breakdown)
    "Speaking Rate": "Words Per Minute — how fast the speaker talks. Deliberate orators: 100–130. Conversational: 140–170. Fast readers/debaters: 180+.",
    "Pitch Variation": "Pitch Standard Deviation — how much the voice pitch varies. High variation = expressive, dynamic delivery. Low = monotone.",
    "Pitch Mean": "Average fundamental frequency (F0) of the voice in Hertz. Typical male: 85–180 Hz. Typical female: 165–255 Hz.",
    "Voice Quality (HNR)": "Harmonics-to-Noise Ratio — how clear and resonant the voice sounds. Higher = cleaner. 20+ dB is excellent; below 7 dB suggests significant hoarseness.",
    "Jitter": "Cycle-to-cycle variation in vocal pitch. Low (<1%) = steady, healthy voice. High (>2%) may indicate vocal strain, aging, or neurological conditions.",
    "Shimmer": "Cycle-to-cycle variation in vocal loudness. Low (<5%) = smooth voice. High values suggest breathiness, hoarseness, or reduced vocal control.",
    "Vocal Fry": "Ratio of creaky, low-frequency vocal pulses. Common in casual modern speech (20–50%). Rare in trained speakers (<5%). Not inherently bad — it's a style marker.",
    "Lexical Diversity": "Ratio of unique words to total words (type-token ratio). Higher = richer vocabulary. Gehrig's farewell: 0.84. Typical speech: 0.40–0.65.",
    "Syntactic Complexity": "Average parse-tree depth of sentences. Higher = more embedded clauses and complex grammar. Academic: 15+. Casual: 8–12.",
    "Grade Level (FK)": "Flesch-Kincaid Grade Level — the U.S. school grade needed to understand the text. Grade 6 = easy for most. Grade 12+ = college-level.",
    "Repetition Score": "Rhetorical repetition score — measures deliberate reuse of phrases for emphasis. MLK's 'I Have a Dream': 24.5. Most speakers: 0–3.",
    "Signal-to-Noise": "Signal-to-Noise Ratio in decibels — how much louder the speech is versus background noise. 30+ dB = studio quality. <10 dB = very noisy.",
    "Articulation Rate": "Speech rate excluding pauses — measures how quickly words are actually produced during active speaking segments.",
    "Discourse Connectedness": "How well ideas flow from one sentence to the next, measured by semantic similarity between adjacent sentences.",
    "Pitch Range": "The span between lowest and highest pitch used. Wider range = more vocal expression. Narrow range can sound flat or robotic.",
    "Vowel Space Index": "How distinctly vowels are articulated, based on formant frequencies. Larger vowel space = clearer pronunciation.",
    "Fluency Score": "Overall fluency rating combining pause patterns, hesitations, and flow. Higher = smoother, more connected speech.",
    "Reading Ease": "Flesch Reading Ease — how easy the text is to read. 90–100 = very easy (5th grade). 30–50 = college level. 0–30 = very difficult.",
}

TEMPLATE_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

router = APIRouter()

_db = None


def get_db() -> SpeechDB:
    global _db
    if _db is None:
        _db = SpeechDB()
    return _db


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard with stats overview."""
    db = get_db()
    stats = db.stats()
    recent = db.list_speeches(limit=10)

    # Get metric distributions for charts
    rows = db.conn.execute("""
        SELECT s.id, s.title, s.speaker, s.category, s.products,
               a.wpm, a.pitch_mean_hz, a.vocal_fry_ratio,
               a.lexical_diversity, a.repetition_score, a.quality_level
        FROM speeches s
        LEFT JOIN analyses a ON a.speech_id = s.id
        ORDER BY s.id
    """).fetchall()

    speeches_data = []
    for r in rows:
        speeches_data.append({
            "id": r["id"],
            "title": r["title"],
            "speaker": r["speaker"] or "Unknown",
            "category": r["category"] or "",
            "wpm": r["wpm"],
            "pitch": r["pitch_mean_hz"],
            "fry": r["vocal_fry_ratio"],
            "lexdiv": r["lexical_diversity"],
            "rep": r["repetition_score"],
            "quality": r["quality_level"],
        })

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "stats": stats,
        "recent": recent,
        "speeches_data": json.dumps(speeches_data),
        "metric_info": METRIC_EXPLANATIONS,
    })


@router.get("/speeches", response_class=HTMLResponse)
async def speech_list(
    request: Request,
    category: str = Query(None),
    product: str = Query(None),
    q: str = Query(None),
):
    """Browse all speeches with filters."""
    db = get_db()
    speeches = db.list_speeches(limit=500, category=category, product=product)

    # Text search filter
    if q:
        q_lower = q.lower()
        speeches = [
            s for s in speeches
            if q_lower in (s["title"] or "").lower()
            or q_lower in (s["speaker"] or "").lower()
            or q_lower in (s.get("description") or "").lower()
        ]

    # Get metrics for each speech
    for s in speeches:
        analysis = db.get_analysis(s["id"])
        s["metrics"] = {}
        if analysis:
            s["metrics"] = {
                k: analysis[k] for k in [
                    "wpm", "pitch_mean_hz", "vocal_fry_ratio",
                    "lexical_diversity", "quality_level", "sentiment",
                ] if analysis.get(k) is not None
            }

    # Get all categories and products for filter dropdowns
    all_categories = sorted(set(
        s["category"] for s in db.list_speeches(limit=500) if s.get("category")
    ))
    all_products = sorted(db.stats().get("products", {}).keys())

    return templates.TemplateResponse("speeches.html", {
        "request": request,
        "speeches": speeches,
        "categories": all_categories,
        "products": all_products,
        "current_category": category,
        "current_product": product,
        "current_q": q or "",
        "total": len(speeches),
    })


@router.get("/speeches/{speech_id}", response_class=HTMLResponse)
async def speech_detail(
    request: Request,
    speech_id: int,
    profile: str = Query("general"),
):
    """Detailed view of a single speech analysis."""
    db = get_db()
    speech = db.get_speech(speech_id)
    if not speech:
        return HTMLResponse("<h1>Speech not found</h1>", status_code=404)

    analysis = db.get_analysis(speech_id)
    transcription = db.get_transcription(speech_id)
    audio = db.get_audio(speech_id)
    spectrogram = db.get_spectrogram(speech_id, "combined")
    full_result = db.get_full_analysis(speech_id)

    # Get scoring data
    if profile not in SCORING_PROFILES:
        profile = "general"
    score_data = score_speech(db, speech_id, profile) if analysis else None
    available_profiles = list(SCORING_PROFILES.keys())

    return templates.TemplateResponse("speech_detail.html", {
        "request": request,
        "speech": speech,
        "analysis": analysis,
        "transcription": transcription,
        "has_audio": audio is not None,
        "audio_size": audio["size_bytes"] if audio else 0,
        "has_spectrogram": spectrogram is not None,
        "full_result": json.dumps(full_result, indent=2) if full_result else "{}",
        "score": score_data,
        "score_json": json.dumps(score_data) if score_data else "null",
        "current_profile": profile,
        "available_profiles": available_profiles,
        "metric_info": METRIC_EXPLANATIONS,
    })


@router.get("/speeches/{speech_id}/spectrogram.png")
async def frontend_spectrogram(speech_id: int):
    """Serve spectrogram image for frontend (no auth required)."""
    from fastapi.responses import Response
    db = get_db()
    spec = db.get_spectrogram(speech_id, spec_type="combined")
    if not spec:
        # Try any type
        for t in ("spectrogram", "mel", "loudness"):
            spec = db.get_spectrogram(speech_id, spec_type=t)
            if spec:
                break
    if not spec:
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse("No spectrogram", status_code=404)
    fmt = spec.get("format", "png")
    mime = {"png": "image/png", "jpg": "image/jpeg"}.get(fmt, "image/png")
    return Response(content=spec["data"], media_type=mime,
                    headers={"Cache-Control": "public, max-age=3600"})


@router.get("/speeches/{speech_id}/audio.opus")
async def frontend_audio(speech_id: int):
    """Serve audio for frontend player (no auth required)."""
    from fastapi.responses import Response
    db = get_db()
    audio = db.get_audio(speech_id)
    if not audio:
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse("No audio", status_code=404)
    return Response(content=audio["data"], media_type="audio/opus",
                    headers={"Cache-Control": "public, max-age=3600"})


@router.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Upload page for new audio analysis."""
    return templates.TemplateResponse("upload.html", {
        "request": request,
    })


@router.get("/methodology", response_class=HTMLResponse)
async def methodology_page(request: Request):
    """Scoring methodology and metrics documentation."""
    from scoring import METRIC_DEFS, SCORING_PROFILES
    db = get_db()
    stats = db.stats()
    return templates.TemplateResponse("methodology.html", {
        "request": request,
        "metric_defs": METRIC_DEFS,
        "metric_info": METRIC_EXPLANATIONS,
        "profiles": SCORING_PROFILES,
        "total_speeches": stats["speeches"],
        "total_analyses": stats["analyses"],
    })


@router.get("/compare", response_class=HTMLResponse)
async def compare_page(
    request: Request,
    ids: str = Query(None),
    profile: str = Query("general"),
):
    """Side-by-side speech comparison with scoring."""
    db = get_db()
    speeches = db.list_speeches(limit=500)

    if profile not in SCORING_PROFILES:
        profile = "general"

    compared = []
    if ids:
        id_list = [int(x.strip()) for x in ids.split(",") if x.strip().isdigit()]
        for sid in id_list[:4]:  # max 4 comparisons
            speech = db.get_speech(sid)
            if speech:
                analysis = db.get_analysis(sid)
                transcription = db.get_transcription(sid)
                score_data = score_speech(db, sid, profile) if analysis else None
                compared.append({
                    "speech": speech,
                    "analysis": analysis,
                    "transcription": transcription,
                    "score": score_data,
                })

    return templates.TemplateResponse("compare.html", {
        "request": request,
        "speeches": speeches,
        "compared": compared,
        "metric_info": METRIC_EXPLANATIONS,
        "compared_json": json.dumps([
            {
                "id": c["speech"]["id"],
                "title": c["speech"]["title"],
                "score": c.get("score"),
                "metrics": {
                    k: c["analysis"][k] for k in [
                        "wpm", "pitch_mean_hz", "pitch_std_hz", "jitter_pct", "shimmer_pct",
                        "hnr_db", "vocal_fry_ratio", "lexical_diversity", "syntactic_complexity",
                        "fk_grade_level", "repetition_score",
                    ] if c.get("analysis") and c["analysis"].get(k) is not None
                } if c.get("analysis") else {},
            }
            for c in compared
        ]) if compared else "[]",
        "current_ids": ids or "",
        "current_profile": profile,
        "available_profiles": list(SCORING_PROFILES.keys()),
    })
