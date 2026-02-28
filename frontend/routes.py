"""Frontend page routes — serves Jinja2 templates."""

import os
import sys
import json
import logging
from pathlib import Path

from fastapi import APIRouter, Request, Query, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates

SPEECH3_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SPEECH3_DIR not in sys.path:
    sys.path.insert(0, SPEECH3_DIR)

from speech_db import SpeechDB
from scoring import score_speech, benchmark_speech, SCORING_PROFILES
from .auth import (
    get_current_user, require_auth, create_session, destroy_session,
    verify_google_token, is_email_allowed, SESSION_COOKIE, ALLOWED_TIERS
)

logger = logging.getLogger("speechscore.frontend")


def is_admin_user(request: Request) -> bool:
    """Check if the current user has admin tier."""
    user = get_current_user(request)
    if not user:
        return False
    email = user.get("email", "")
    db = get_db()
    row = db.conn.execute("SELECT tier FROM users WHERE LOWER(email) = LOWER(?)", (email,)).fetchone()
    if not row:
        return False
    return (row[0] or "free").lower() == "admin"


def get_user_tier(request: Request) -> str:
    """Get the current user's tier."""
    user = get_current_user(request)
    if not user:
        return "none"
    email = user.get("email", "")
    db = get_db()
    row = db.conn.execute("SELECT tier FROM users WHERE LOWER(email) = LOWER(?)", (email,)).fetchone()
    if not row:
        return "none"
    return (row[0] or "free").lower()

# Google OAuth client ID (same as API)
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "819724788424-r7p1tr8l7k2b2glsdn1ktj0o6k7cspb0.apps.googleusercontent.com")

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
    
    # Pronunciation metrics
    "Pronunciation Clarity": "Mean confidence score across all words from Wav2Vec2 acoustic model. 90%+ = excellent clarity. 70-90% = good. Below 70% = needs work.",
    "Pronunciation Consistency": "Standard deviation of per-word confidence scores. Lower = more consistent pronunciation. High variance suggests some words are clear while others are problematic.",
    "Pronunciation Accuracy": "Percentage of words scoring below the 70% confidence threshold. Lower = better. Identifies proportion of potentially mispronounced words.",
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


# ---------------------------------------------------------------------------
# Authentication Routes
# ---------------------------------------------------------------------------

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = Query(None)):
    """Login page with Google Sign-In."""
    # If already logged in, redirect to dashboard
    user = get_current_user(request)
    if user:
        return RedirectResponse(url="/dashboard", status_code=302)
    
    # Get Google client ID from config
    import json
    import os
    oauth_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "api", "oauth_config.json")
    google_client_id = ""
    if os.path.exists(oauth_config_path):
        with open(oauth_config_path) as f:
            oauth_config = json.load(f)
            client_ids = oauth_config.get("google", {}).get("client_ids", [])
            if client_ids:
                google_client_id = client_ids[0]
    
    return templates.TemplateResponse("login.html", {
        "request": request,
        "error": error,
        "google_client_id": google_client_id,
    })


@router.post("/auth/google/callback")
async def google_callback(request: Request, credential: str = Form(...)):
    """Handle Google Sign-In callback."""
    # Verify the Google token
    user_info = await verify_google_token(credential)
    
    if not user_info or not user_info.get("email"):
        return JSONResponse(
            status_code=401,
            content={"error": "Invalid Google credential"}
        )
    
    email = user_info["email"]
    
    # Check if email is allowed
    if not is_email_allowed(email):
        logger.warning(f"Unauthorized login attempt from: {email}")
        return RedirectResponse(
            url="/?error=Access+denied.+Your+email+is+not+authorized.#login",
            status_code=302
        )
    
    # Create session
    session_token = create_session(email)
    
    # Redirect to dashboard with session cookie
    response = RedirectResponse(url="/dashboard", status_code=302)
    response.set_cookie(
        key=SESSION_COOKIE,
        value=session_token,
        httponly=True,
        secure=True,  # Only send over HTTPS
        samesite="lax",
        max_age=60 * 60 * 24 * 7,  # 1 week
    )
    
    logger.info(f"User logged in: {email}")
    return response


@router.get("/logout")
async def logout(request: Request):
    """Log out and clear session."""
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        destroy_session(token)
    
    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie(SESSION_COOKIE)
    return response


# ---------------------------------------------------------------------------
# Protected Dashboard Routes
# ---------------------------------------------------------------------------

@router.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    """Landing page — redirects to login if not authenticated, dashboard if authenticated."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    return RedirectResponse(url="/dashboard", status_code=302)


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard with stats overview (requires auth)."""
    # Require authentication
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    
    db = get_db()
    stats = db.stats()
    recent = db.list_speeches(limit=10)

    # Get database-wide metric averages
    avg_metrics = db.conn.execute("""
        SELECT 
            COUNT(*) as total_analyzed,
            ROUND(AVG(wpm), 1) as avg_wpm,
            ROUND(AVG(articulation_rate), 1) as avg_articulation_rate,
            ROUND(AVG(pitch_mean_hz), 1) as avg_pitch_hz,
            ROUND(AVG(pitch_std_hz), 1) as avg_pitch_std,
            ROUND(AVG(pitch_range_hz), 1) as avg_pitch_range,
            ROUND(AVG(jitter_pct), 2) as avg_jitter_pct,
            ROUND(AVG(shimmer_pct), 2) as avg_shimmer_pct,
            ROUND(AVG(hnr_db), 1) as avg_hnr_db,
            ROUND(AVG(vocal_fry_ratio) * 100, 1) as avg_vocal_fry_pct,
            ROUND(AVG(uptalk_ratio) * 100, 1) as avg_uptalk_pct,
            ROUND(AVG(rate_variability_cv), 2) as avg_rate_var,
            ROUND(AVG(lexical_diversity), 2) as avg_lex_diversity,
            ROUND(AVG(fk_grade_level), 1) as avg_fk_grade,
            ROUND(AVG(flesch_reading_ease), 1) as avg_flesch_ease,
            ROUND(AVG(snr_db), 1) as avg_snr_db
        FROM analyses
        WHERE quality_level NOT IN ('UNUSABLE', '')
    """).fetchone()
    
    # Quality distribution
    quality_dist = db.conn.execute("""
        SELECT quality_level, COUNT(*) as count
        FROM analyses
        WHERE quality_level IS NOT NULL AND quality_level != ''
        GROUP BY quality_level
        ORDER BY count DESC
    """).fetchall()

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
        "avg_metrics": dict(avg_metrics) if avg_metrics else {},
        "quality_dist": [{"level": q["quality_level"], "count": q["count"]} for q in quality_dist],
    })


@router.get("/speeches", response_class=HTMLResponse)
async def speech_list_redirect(
    request: Request,
    category: str = Query(None),
    product: str = Query(None),
    q: str = Query(None),
):
    """Redirect old /speeches URL to /recordings."""
    params = []
    if category:
        params.append(f"category={category}")
    if product:
        params.append(f"product={product}")
    if q:
        params.append(f"q={q}")
    url = "/recordings" + ("?" + "&".join(params) if params else "")
    return RedirectResponse(url=url, status_code=301)


@router.get("/recordings", response_class=HTMLResponse)
async def recording_list(
    request: Request,
    category: str = Query(None),
    product: str = Query(None),
    q: str = Query(None),
):
    """Browse all recordings with filters."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/", status_code=302)
    
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
async def speech_detail_redirect(
    request: Request,
    speech_id: int,
    profile: str = Query("general"),
):
    """Redirect old /speeches/{id} URL to /recordings/{id}."""
    url = f"/recordings/{speech_id}"
    if profile != "general":
        url += f"?profile={profile}"
    return RedirectResponse(url=url, status_code=301)


@router.get("/recordings/{speech_id}", response_class=HTMLResponse)
async def recording_detail(
    request: Request,
    speech_id: int,
    profile: str = Query("general"),
):
    """Detailed view of a single recording analysis."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/", status_code=302)
    
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
@router.get("/recordings/{speech_id}/spectrogram.png")
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
@router.get("/recordings/{speech_id}/audio.opus")
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
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/", status_code=302)
    
    return templates.TemplateResponse("upload.html", {
        "request": request,
    })


@router.get("/methodology", response_class=HTMLResponse)
async def methodology_page(request: Request):
    """Scoring methodology and metrics documentation."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/", status_code=302)
    
    # Check if user is admin (for pipeline internals section)
    is_admin = is_admin_user(request)
    
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
        "is_admin": is_admin,
    })


@router.get("/compare", response_class=HTMLResponse)
async def compare_page(
    request: Request,
    ids: str = Query(None),
    profile: str = Query("general"),
):
    """Side-by-side speech comparison with scoring."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/", status_code=302)
    
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


@router.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    """Admin dashboard with platform statistics. Restricted to admin tier users."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/", status_code=302)
    
    # Only allow admin tier
    if not is_admin_user(request):
        return HTMLResponse("<h1>Access Denied</h1><p>Admin access is restricted to admin tier users.</p>", status_code=403)
    
    db = get_db()
    from datetime import datetime, timedelta
    
    # Basic stats
    total_users = db.conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    total_recordings = db.conn.execute("SELECT COUNT(*) FROM speeches WHERE user_id IS NOT NULL").fetchone()[0]
    
    # Active users (7 and 30 days)
    now = datetime.utcnow()
    week_ago = (now - timedelta(days=7)).isoformat()
    month_ago = (now - timedelta(days=30)).isoformat()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    week_start = (now - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    
    active_7d = db.conn.execute("""
        SELECT COUNT(DISTINCT user_id) FROM speeches 
        WHERE user_id IS NOT NULL AND created_at >= ?
    """, (week_ago,)).fetchone()[0]
    
    active_30d = db.conn.execute("""
        SELECT COUNT(DISTINCT user_id) FROM speeches 
        WHERE user_id IS NOT NULL AND created_at >= ?
    """, (month_ago,)).fetchone()[0]
    
    # Recordings today and this week
    recordings_today = db.conn.execute("""
        SELECT COUNT(*) FROM speeches WHERE user_id IS NOT NULL AND created_at >= ?
    """, (today_start,)).fetchone()[0]
    
    recordings_week = db.conn.execute("""
        SELECT COUNT(*) FROM speeches WHERE user_id IS NOT NULL AND created_at >= ?
    """, (week_start,)).fetchone()[0]
    
    # Storage (audio files)
    audio_dir = Path("/home/jsuppe/speech3/audio")
    try:
        storage_bytes = sum(f.stat().st_size for f in audio_dir.rglob("*") if f.is_file()) if audio_dir.exists() else 0
    except:
        storage_bytes = 0
    storage_mb = storage_bytes / (1024 * 1024)
    
    # Average score (from analyses with scores)
    avg_score_row = db.conn.execute("""
        SELECT AVG(
            (COALESCE(lexical_diversity, 0.5) * 20) +
            (COALESCE(hnr_db, 15) / 25 * 20) +
            (MIN(MAX(wpm, 100), 160) - 100) / 60 * 20 +
            (20 - MIN(COALESCE(fk_grade_level, 10), 20)) +
            (COALESCE(pitch_std_hz, 30) / 60 * 20)
        ) as avg_score
        FROM analyses WHERE wpm IS NOT NULL
    """).fetchone()
    avg_score = avg_score_row[0] if avg_score_row and avg_score_row[0] else None
    
    # Top users by recordings
    top_users_rows = db.conn.execute("""
        SELECT u.*, COUNT(s.id) as speech_count,
               AVG(CASE WHEN a.wpm IS NOT NULL THEN
                   (COALESCE(a.lexical_diversity, 0.5) * 20) +
                   (COALESCE(a.hnr_db, 15) / 25 * 20) +
                   (MIN(MAX(a.wpm, 100), 160) - 100) / 60 * 20
               END) as avg_score
        FROM users u
        LEFT JOIN speeches s ON s.user_id = u.id
        LEFT JOIN analyses a ON a.speech_id = s.id
        GROUP BY u.id
        ORDER BY speech_count DESC
        LIMIT 10
    """).fetchall()
    top_users = [dict(r) for r in top_users_rows]
    
    # Recent recordings with user info
    recent_rows = db.conn.execute("""
        SELECT s.*, u.name as user_name,
               (COALESCE(a.lexical_diversity, 0.5) * 20) +
               (COALESCE(a.hnr_db, 15) / 25 * 20) +
               (MIN(MAX(a.wpm, 100), 160) - 100) / 60 * 20 as score
        FROM speeches s
        LEFT JOIN users u ON u.id = s.user_id
        LEFT JOIN analyses a ON a.speech_id = s.id
        WHERE s.user_id IS NOT NULL
        ORDER BY s.created_at DESC
        LIMIT 15
    """).fetchall()
    recent_recordings = [dict(r) for r in recent_rows]
    
    # Chart data: recordings by day (last 30 days)
    recordings_by_day = db.conn.execute("""
        SELECT DATE(created_at) as date, COUNT(*) as count
        FROM speeches
        WHERE user_id IS NOT NULL AND created_at >= ?
        GROUP BY DATE(created_at)
        ORDER BY date
    """, (month_ago,)).fetchall()
    
    # Chart data: user signups by day (last 30 days)
    users_by_day = db.conn.execute("""
        SELECT DATE(created_at) as date, COUNT(*) as count
        FROM users
        WHERE created_at >= ?
        GROUP BY DATE(created_at)
        ORDER BY date
    """, (month_ago,)).fetchall()
    
    # Score distribution
    score_dist = [0, 0, 0, 0, 0]  # 0-20, 20-40, 40-60, 60-80, 80-100
    score_rows = db.conn.execute("""
        SELECT (COALESCE(lexical_diversity, 0.5) * 20) +
               (COALESCE(hnr_db, 15) / 25 * 20) +
               (MIN(MAX(wpm, 100), 160) - 100) / 60 * 20 as score
        FROM analyses WHERE wpm IS NOT NULL
    """).fetchall()
    for row in score_rows:
        if row[0] is not None:
            idx = min(int(row[0] / 20), 4)
            score_dist[idx] += 1
    
    # Category distribution
    categories = db.conn.execute("""
        SELECT COALESCE(category, 'uncategorized') as category, COUNT(*) as count
        FROM speeches
        WHERE user_id IS NOT NULL
        GROUP BY category
        ORDER BY count DESC
    """).fetchall()
    
    # Survey stats
    survey_total = db.conn.execute("SELECT COUNT(*) FROM user_surveys").fetchone()[0]
    survey_completion_rate = (survey_total / total_users * 100) if total_users > 0 else 0
    
    # Survey breakdown by primary goal
    survey_goals = db.conn.execute("""
        SELECT primary_goal, COUNT(*) as count
        FROM user_surveys
        GROUP BY primary_goal
        ORDER BY count DESC
    """).fetchall()
    
    # Survey breakdown by skill level
    survey_skill_levels = db.conn.execute("""
        SELECT skill_level, COUNT(*) as count
        FROM user_surveys
        GROUP BY skill_level
        ORDER BY count DESC
    """).fetchall()
    
    # Survey breakdown by context
    survey_contexts = db.conn.execute("""
        SELECT context, COUNT(*) as count
        FROM user_surveys
        GROUP BY context
        ORDER BY count DESC
    """).fetchall()
    
    # Focus areas (need to parse JSON array)
    focus_areas_raw = db.conn.execute("SELECT focus_areas FROM user_surveys").fetchall()
    focus_area_counts = {}
    for row in focus_areas_raw:
        try:
            areas = json.loads(row[0])
            for area in areas:
                focus_area_counts[area] = focus_area_counts.get(area, 0) + 1
        except:
            pass
    focus_areas_sorted = sorted(focus_area_counts.items(), key=lambda x: -x[1])
    
    # Recent survey submissions
    recent_surveys = db.conn.execute("""
        SELECT us.*, u.name, u.email
        FROM user_surveys us
        JOIN users u ON us.user_id = u.id
        ORDER BY us.completed_at DESC
        LIMIT 10
    """).fetchall()
    recent_surveys = [dict(r) for r in recent_surveys]
    
    chart_data = json.dumps({
        "recordings_by_day": [{"date": r[0], "count": r[1]} for r in recordings_by_day],
        "users_by_day": [{"date": r[0], "count": r[1]} for r in users_by_day],
        "score_distribution": score_dist,
        "categories": [{"category": c[0], "count": c[1]} for c in categories],
        "survey_goals": [{"goal": g[0], "count": g[1]} for g in survey_goals],
        "survey_skill_levels": [{"level": s[0], "count": s[1]} for s in survey_skill_levels],
        "survey_contexts": [{"context": c[0], "count": c[1]} for c in survey_contexts],
        "survey_focus_areas": [{"area": a[0], "count": a[1]} for a in focus_areas_sorted],
    })
    
    # Get all users for the full users table
    all_users_rows = db.conn.execute("""
        SELECT u.*, COUNT(s.id) as speech_count
        FROM users u
        LEFT JOIN speeches s ON s.user_id = u.id
        GROUP BY u.id
        ORDER BY u.created_at DESC
    """).fetchall()
    all_users = [dict(r) for r in all_users_rows]
    
    return templates.TemplateResponse("admin.html", {
        "request": request,
        "stats": {
            "total_users": total_users,
            "total_recordings": total_recordings,
            "active_7d": active_7d,
            "active_30d": active_30d,
            "recordings_today": recordings_today,
            "recordings_week": recordings_week,
            "storage_mb": storage_mb,
            "avg_score": avg_score,
            "survey_total": survey_total,
            "survey_completion_rate": survey_completion_rate,
        },
        "top_users": top_users,
        "all_users": all_users,
        "recent_recordings": recent_recordings,
        "recent_surveys": recent_surveys,
        "chart_data": chart_data,
    })


@router.get("/users", response_class=HTMLResponse)
async def users_page(request: Request, page: int = Query(1, ge=1), q: str = Query(None)):
    """List all users with pagination and search (admin only)."""
    current_user = get_current_user(request)
    if not current_user:
        return RedirectResponse(url="/", status_code=302)
    
    # Only allow Jon
    if not is_admin_user(request):
        return HTMLResponse("<h1>Access Denied</h1><p>Admin access is restricted.</p>", status_code=403)
    
    db = get_db()
    per_page = 20
    offset = (page - 1) * per_page
    
    # Build query with optional search
    if q:
        search_query = f"%{q}%"
        count_row = db.conn.execute(
            "SELECT COUNT(*) FROM users WHERE name LIKE ? OR email LIKE ?",
            (search_query, search_query)
        ).fetchone()
        total = count_row[0]
        
        rows = db.conn.execute("""
            SELECT u.*, 
                   (SELECT COUNT(*) FROM speeches WHERE user_id = u.id) as speech_count
            FROM users u
            WHERE u.name LIKE ? OR u.email LIKE ?
            ORDER BY u.created_at DESC
            LIMIT ? OFFSET ?
        """, (search_query, search_query, per_page, offset)).fetchall()
    else:
        count_row = db.conn.execute("SELECT COUNT(*) FROM users").fetchone()
        total = count_row[0]
        
        rows = db.conn.execute("""
            SELECT u.*, 
                   (SELECT COUNT(*) FROM speeches WHERE user_id = u.id) as speech_count
            FROM users u
            ORDER BY u.created_at DESC
            LIMIT ? OFFSET ?
        """, (per_page, offset)).fetchall()
    
    users = [dict(r) for r in rows]
    total_pages = (total + per_page - 1) // per_page
    
    return templates.TemplateResponse("users.html", {
        "request": request,
        "users": users,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
        "search_query": q,
    })


@router.get("/admin/users/{user_id}", response_class=HTMLResponse)
async def admin_user_detail(request: Request, user_id: int, page: int = Query(1, ge=1)):
    """View details for a specific user (admin only)."""
    current_user = get_current_user(request)
    if not current_user:
        return RedirectResponse(url="/", status_code=302)
    
    # Only allow Jon
    if not is_admin_user(request):
        return HTMLResponse("<h1>Access Denied</h1><p>Admin access is restricted.</p>", status_code=403)
    
    db = get_db()
    
    # Get user info
    row = db.conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not row:
        return HTMLResponse("<h1>User not found</h1>", status_code=404)
    
    user = dict(row)
    
    # Pagination for speeches
    per_page = 10
    offset = (page - 1) * per_page
    
    # Get total speech count
    count_row = db.conn.execute(
        "SELECT COUNT(*) FROM speeches WHERE user_id = ?", (user_id,)
    ).fetchone()
    total_speeches = count_row[0]
    total_pages = (total_speeches + per_page - 1) // per_page if total_speeches > 0 else 1
    
    # Get user's speeches with analysis data (paginated)
    speech_rows = db.conn.execute("""
        SELECT s.*, a.quality_level, a.wpm
        FROM speeches s
        LEFT JOIN analyses a ON a.speech_id = s.id
        WHERE s.user_id = ?
        ORDER BY s.created_at DESC
        LIMIT ? OFFSET ?
    """, (user_id, per_page, offset)).fetchall()
    speeches = [dict(r) for r in speech_rows]
    
    # Get coach messages
    coach_messages = db.get_coach_messages(user_id, limit=50)
    coach_stats = db.get_user_coach_stats(user_id)
    
    return templates.TemplateResponse("user_detail.html", {
        "request": request,
        "user": user,
        "speeches": speeches,
        "total_speeches": total_speeches,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
        "coach_messages": coach_messages,
        "coach_stats": coach_stats,
        "is_admin": True,
    })


@router.post("/admin/users/{user_id}/recalc-usage")
async def admin_recalc_user_usage(request: Request, user_id: int):
    """Recalculate a user's monthly usage from actual recordings (admin only)."""
    current_user = get_current_user(request)
    if not current_user or not is_admin_user(request):
        return HTMLResponse("<h1>Access Denied</h1>", status_code=403)
    
    from datetime import datetime
    current_month = datetime.utcnow().strftime("%Y-%m")
    month_start = f"{current_month}-01"
    
    db = get_db()
    db.conn.execute("""
        UPDATE users 
        SET monthly_recordings = (
            SELECT COUNT(*) FROM speeches 
            WHERE user_id = ? AND created_at >= ?
        ),
        monthly_audio_minutes = (
            SELECT COALESCE(SUM(duration_sec)/60.0, 0) FROM speeches 
            WHERE user_id = ? AND created_at >= ?
        ),
        usage_month = ?
        WHERE id = ?
    """, (user_id, month_start, user_id, month_start, current_month, user_id))
    db.conn.commit()
    
    return RedirectResponse(url=f"/admin/users/{user_id}", status_code=302)


@router.post("/admin/users/{user_id}/reset-usage")
async def admin_reset_user_usage(request: Request, user_id: int):
    """Reset a user's monthly usage counters to zero (admin only)."""
    current_user = get_current_user(request)
    if not current_user or not is_admin_user(request):
        return HTMLResponse("<h1>Access Denied</h1>", status_code=403)
    
    db = get_db()
    db.conn.execute("""
        UPDATE users 
        SET monthly_recordings = 0, monthly_audio_minutes = 0.0 
        WHERE id = ?
    """, (user_id,))
    db.conn.commit()
    
    return RedirectResponse(url=f"/admin/users/{user_id}", status_code=302)


@router.post("/admin/users/{user_id}/set-tier")
async def admin_set_user_tier(request: Request, user_id: int, tier: str = Form(...)):
    """Change a user's tier (admin only)."""
    current_user = get_current_user(request)
    if not current_user or not is_admin_user(request):
        return HTMLResponse("<h1>Access Denied</h1>", status_code=403)
    
    # Validate tier
    valid_tiers = ['free', 'pro', 'enterprise', 'admin']
    if tier not in valid_tiers:
        return HTMLResponse(f"<h1>Invalid tier: {tier}</h1>", status_code=400)
    
    db = get_db()
    db.conn.execute("UPDATE users SET tier = ? WHERE id = ?", (tier, user_id))
    db.conn.commit()
    
    return RedirectResponse(url=f"/admin/users/{user_id}", status_code=302)


@router.post("/admin/users/{user_id}/toggle-enabled")
async def admin_toggle_user_enabled(request: Request, user_id: int):
    """Toggle a user's enabled status (admin only)."""
    current_user = get_current_user(request)
    if not current_user or not is_admin_user(request):
        return HTMLResponse("<h1>Access Denied</h1>", status_code=403)
    
    # Don't allow disabling yourself
    db = get_db()
    row = db.conn.execute("SELECT email, enabled FROM users WHERE id = ?", (user_id,)).fetchone()
    if row and row[0].lower() == current_user.get("email", "").lower():
        return HTMLResponse("<h1>Cannot disable your own account</h1>", status_code=400)
    
    # Toggle enabled status (default to 1 if null)
    current_enabled = row[1] if row and row[1] is not None else 1
    new_enabled = 0 if current_enabled else 1
    db.conn.execute("UPDATE users SET enabled = ? WHERE id = ?", (new_enabled, user_id))
    db.conn.commit()
    
    return RedirectResponse(url=f"/admin/users/{user_id}", status_code=302)


@router.post("/admin/users/{user_id}/delete")
async def admin_delete_user(request: Request, user_id: int):
    """Delete a user and all their data (admin only)."""
    current_user = get_current_user(request)
    if not current_user or not is_admin_user(request):
        return HTMLResponse("<h1>Access Denied</h1>", status_code=403)
    
    # Don't allow deleting yourself
    db = get_db()
    row = db.conn.execute("SELECT email FROM users WHERE id = ?", (user_id,)).fetchone()
    if row and row[0].lower() == current_user.get("email", "").lower():
        return HTMLResponse("<h1>Cannot delete your own account</h1>", status_code=400)
    
    # Delete user (CASCADE should handle speeches, analyses, etc.)
    db.conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    db.conn.commit()
    
    return RedirectResponse(url="/admin", status_code=302)


@router.get("/users/{user_id}", response_class=HTMLResponse)
async def users_detail_redirect(request: Request, user_id: int):
    """Redirect old user detail URLs to admin section."""
    return RedirectResponse(url=f"/admin/users/{user_id}", status_code=302)


# ---------------------------------------------------------------------------
# Frontend API Endpoints (use session auth instead of API auth)
# ---------------------------------------------------------------------------

@router.patch("/api/speeches/{speech_id}")
async def frontend_update_speech(
    request: Request,
    speech_id: int,
    title: str = Form(None),
    speaker: str = Form(None),
):
    """Update speech metadata from frontend (uses session auth)."""
    user = get_current_user(request)
    if not user:
        return JSONResponse(
            status_code=401,
            content={"error": {"code": "UNAUTHORIZED", "message": "Not logged in"}}
        )
    
    # Parse JSON body if Content-Type is application/json
    body = {}
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            title = body.get("title", title)
            speaker = body.get("speaker", speaker)
        except:
            pass
    
    if title is None and speaker is None:
        return JSONResponse(
            status_code=400,
            content={"error": {"code": "NO_CHANGES", "message": "No fields to update"}}
        )
    
    db = get_db()
    
    # Get the speech
    speech = db.get_speech(speech_id)
    if not speech:
        return JSONResponse(
            status_code=404,
            content={"error": {"code": "NOT_FOUND", "message": "Speech not found"}}
        )
    
    # Update fields
    updates = []
    params = []
    if title is not None:
        updates.append("title = ?")
        params.append(title)
    if speaker is not None:
        updates.append("speaker = ?")
        params.append(speaker)
    
    params.append(speech_id)
    
    db.conn.execute(
        f"UPDATE speeches SET {', '.join(updates)} WHERE id = ?",
        params
    )
    db.conn.commit()
    
    logger.info(f"Updated speech {speech_id}: title={title}, speaker={speaker}")
    
    return JSONResponse(content={
        "success": True,
        "speech_id": speech_id,
        "title": title,
        "speaker": speaker,
    })
