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


def _generate_alerts(
    dau: int = 0,
    error_rate: float = 0,
    cpu: float = 0,
    memory: float = 0,
    gpu: float = 0,
    disk: float = 0,
    r2_connected: bool = True,
    supabase_connected: bool = True,
    churn_rate: float = 0,
    error_cards: int = 0,
) -> list:
    """Generate alerts based on metric thresholds."""
    alerts = []
    
    # Critical alerts (red)
    if not r2_connected:
        alerts.append({"level": "critical", "icon": "🚨", "title": "R2 Storage Disconnected", 
                       "message": "Cloudflare R2 is unavailable. New uploads will fail."})
    if not supabase_connected:
        alerts.append({"level": "critical", "icon": "🚨", "title": "Supabase Disconnected", 
                       "message": "PostgreSQL database is unavailable."})
    if error_rate > 10:
        alerts.append({"level": "critical", "icon": "🚨", "title": f"High Error Rate: {error_rate:.1f}%", 
                       "message": "API error rate is critically high. Check logs."})
    if disk > 95:
        alerts.append({"level": "critical", "icon": "🚨", "title": f"Disk Almost Full: {disk:.0f}%", 
                       "message": "Server disk space is critically low."})
    if memory > 95:
        alerts.append({"level": "critical", "icon": "🚨", "title": f"Memory Critical: {memory:.0f}%", 
                       "message": "Server memory is critically low."})
    
    # Warning alerts (yellow)
    if error_rate > 5 and error_rate <= 10:
        alerts.append({"level": "warning", "icon": "⚠️", "title": f"Elevated Error Rate: {error_rate:.1f}%", 
                       "message": "API error rate is elevated. Monitor closely."})
    if cpu > 90:
        alerts.append({"level": "warning", "icon": "⚠️", "title": f"High CPU Usage: {cpu:.0f}%", 
                       "message": "Server CPU usage is high."})
    if memory > 85 and memory <= 95:
        alerts.append({"level": "warning", "icon": "⚠️", "title": f"High Memory Usage: {memory:.0f}%", 
                       "message": "Server memory usage is high."})
    if gpu > 90:
        alerts.append({"level": "warning", "icon": "⚠️", "title": f"High GPU Usage: {gpu:.0f}%", 
                       "message": "GPU utilization is high."})
    if disk > 85 and disk <= 95:
        alerts.append({"level": "warning", "icon": "⚠️", "title": f"Disk Space Low: {disk:.0f}%", 
                       "message": "Server disk space is running low."})
    if churn_rate > 20:
        alerts.append({"level": "warning", "icon": "⚠️", "title": f"High Churn Rate: {churn_rate:.1f}%", 
                       "message": "Monthly churn rate is high. Review user feedback."})
    if error_cards >= 5:
        alerts.append({"level": "warning", "icon": "⚠️", "title": f"{error_cards} Open Error Cards", 
                       "message": "Multiple app errors pending resolution."})
    
    # Info alerts (blue)
    if dau == 0:
        alerts.append({"level": "info", "icon": "ℹ️", "title": "No Active Users Today", 
                       "message": "No recordings have been made today."})
    
    return alerts

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


@router.get("/privacy", response_class=HTMLResponse)
async def privacy_policy(request: Request):
    """Privacy Policy page — public, no auth required."""
    return templates.TemplateResponse("privacy.html", {"request": request})


@router.get("/terms", response_class=HTMLResponse)
async def terms_of_service(request: Request):
    """Terms of Service page — public, no auth required."""
    # For now, redirect to privacy until we create a terms page
    return RedirectResponse(url="/privacy", status_code=302)


@router.get("/downloads", response_class=HTMLResponse)
async def downloads_page(request: Request):
    """Downloads page — public, lists all SpeakFit app APKs."""
    return templates.TemplateResponse("downloads.html", {"request": request})


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard with stats overview (requires auth)."""
    from datetime import datetime, timedelta
    
    # Require authentication
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    
    db = get_db()
    
    # Profile to product mapping
    PROFILE_MAP = {
        'general': {'name': 'SpeakFit', 'icon': '🎙️', 'color': 'bg-indigo-500'},
        'pronunciation': {'name': 'Pronounce', 'icon': '🗣️', 'color': 'bg-blue-500'},
        'presentation': {'name': 'Present', 'icon': '🎯', 'color': 'bg-green-500'},
        'reading_fluency': {'name': 'Reader', 'icon': '📖', 'color': 'bg-purple-500'},
        'oratory': {'name': 'Oratory', 'icon': '🏛️', 'color': 'bg-orange-500'},
        'dementia': {'name': 'Memory', 'icon': '🧠', 'color': 'bg-teal-500'},
        'kids': {'name': 'Ollie', 'icon': '🦉', 'color': 'bg-amber-500'},
    }
    
    # Basic stats
    total_users = db.conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    total_speeches = db.conn.execute("SELECT COUNT(*) FROM speeches").fetchone()[0]
    
    # User recordings (exclude batch imports - those with user_id)
    user_recordings = db.conn.execute(
        "SELECT COUNT(*) FROM speeches WHERE user_id IS NOT NULL"
    ).fetchone()[0]
    
    # New users in last 7 days
    new_users_7d = db.conn.execute("""
        SELECT COUNT(*) FROM users 
        WHERE created_at > datetime('now', '-7 days')
    """).fetchone()[0]
    
    # Recordings in last 7 days (user only)
    recordings_7d = db.conn.execute("""
        SELECT COUNT(*) FROM speeches 
        WHERE user_id IS NOT NULL AND created_at > datetime('now', '-7 days')
    """).fetchone()[0]
    
    # Coach stats
    coach_messages = db.conn.execute("SELECT COUNT(*) FROM coach_messages").fetchone()[0]
    coach_conversations = db.conn.execute("""
        SELECT COUNT(DISTINCT user_id) FROM coach_messages
    """).fetchone()[0]
    
    # User minutes (from user recordings only)
    user_minutes = db.conn.execute("""
        SELECT COALESCE(SUM(duration_sec) / 60.0, 0) FROM speeches WHERE user_id IS NOT NULL
    """).fetchone()[0]
    
    # Avg duration and WPM for user recordings
    avg_stats = db.conn.execute("""
        SELECT 
            ROUND(AVG(s.duration_sec), 1) as avg_duration,
            ROUND(AVG(a.wpm), 0) as avg_wpm
        FROM speeches s
        LEFT JOIN analyses a ON a.speech_id = s.id
        WHERE s.user_id IS NOT NULL
    """).fetchone()
    
    # Reading and dementia sessions
    reading_sessions = db.conn.execute("""
        SELECT COUNT(*) FROM speeches WHERE profile = 'reading_fluency' AND user_id IS NOT NULL
    """).fetchone()[0]
    dementia_sessions = db.conn.execute("""
        SELECT COUNT(*) FROM speeches WHERE profile = 'dementia' AND user_id IS NOT NULL
    """).fetchone()[0]
    
    stats = {
        'users': total_users,
        'total_speeches': total_speeches,
        'user_recordings': user_recordings,
        'new_users_7d': new_users_7d,
        'recordings_7d': recordings_7d,
        'coach_conversations': coach_conversations,
        'coach_messages': coach_messages,
        'user_minutes': user_minutes or 0,
        'avg_duration': avg_stats['avg_duration'] or 0,
        'avg_wpm': int(avg_stats['avg_wpm'] or 0),
        'reading_sessions': reading_sessions,
        'dementia_sessions': dementia_sessions,
    }
    
    # Product usage (user recordings only)
    product_rows = db.conn.execute("""
        SELECT COALESCE(profile, 'general') as profile, COUNT(*) as count
        FROM speeches
        WHERE user_id IS NOT NULL
        GROUP BY profile
        ORDER BY count DESC
    """).fetchall()
    
    total_user_recs = max(1, sum(r['count'] for r in product_rows))
    products = []
    for r in product_rows:
        profile = r['profile'] or 'general'
        info = PROFILE_MAP.get(profile, {'name': profile, 'icon': '📊', 'color': 'bg-gray-500'})
        products.append({
            'name': info['name'],
            'icon': info['icon'],
            'color': info['color'],
            'count': r['count'],
            'percent': round(r['count'] / total_user_recs * 100, 1),
        })
    
    # Top users
    top_users_rows = db.conn.execute("""
        SELECT u.id, u.name, u.tier, COUNT(s.id) as recordings
        FROM users u
        LEFT JOIN speeches s ON s.user_id = u.id
        GROUP BY u.id
        ORDER BY recordings DESC
        LIMIT 5
    """).fetchall()
    top_users = [
        {'name': r['name'] or 'Anonymous', 'tier': r['tier'] or 'free', 'recordings': r['recordings']}
        for r in top_users_rows
    ]
    
    # Average scores by product (user recordings only)
    avg_score_rows = db.conn.execute("""
        SELECT s.profile, ROUND(AVG(s.cached_score), 0) as avg_score
        FROM speeches s
        WHERE s.user_id IS NOT NULL AND s.cached_score IS NOT NULL
        GROUP BY s.profile
        ORDER BY avg_score DESC
    """).fetchall()
    avg_scores = []
    for r in avg_score_rows:
        profile = r['profile'] or 'general'
        info = PROFILE_MAP.get(profile, {'name': profile, 'color': 'bg-gray-500'})
        score = int(r['avg_score'] or 0)
        color = 'bg-green-500' if score >= 80 else 'bg-yellow-500' if score >= 60 else 'bg-red-500'
        avg_scores.append({
            'name': info['name'],
            'value': score,
            'color': color,
        })
    
    # Daily activity (last 14 days, user recordings only)
    daily_rows = db.conn.execute("""
        SELECT date(created_at) as date, COUNT(*) as recordings
        FROM speeches
        WHERE user_id IS NOT NULL AND created_at > datetime('now', '-14 days')
        GROUP BY date(created_at)
        ORDER BY date
    """).fetchall()
    daily_activity = [{'date': r['date'][5:], 'recordings': r['recordings']} for r in daily_rows]
    
    # Recent user recordings
    recent_rows = db.conn.execute("""
        SELECT s.id, s.title, s.profile, s.created_at, s.duration_sec,
               u.name as user_name, s.cached_score as overall_score
        FROM speeches s
        LEFT JOIN users u ON u.id = s.user_id
        WHERE s.user_id IS NOT NULL
        ORDER BY s.created_at DESC
        LIMIT 10
    """).fetchall()
    
    def time_ago(dt_str):
        if not dt_str:
            return '—'
        try:
            dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
            diff = datetime.now(dt.tzinfo) - dt if dt.tzinfo else datetime.now() - dt
            if diff.days > 0:
                return f"{diff.days}d ago"
            elif diff.seconds > 3600:
                return f"{diff.seconds // 3600}h ago"
            elif diff.seconds > 60:
                return f"{diff.seconds // 60}m ago"
            return "just now"
        except:
            return dt_str[:10]
    
    recent_user_recordings = []
    for r in recent_rows:
        profile = r['profile'] or 'general'
        info = PROFILE_MAP.get(profile, {'name': profile, 'icon': '📊', 'color': 'bg-gray-500'})
        score = r['overall_score']
        score_color = 'text-green-400' if score and score >= 80 else 'text-yellow-400' if score and score >= 60 else 'text-red-400' if score else 'text-gray-600'
        recent_user_recordings.append({
            'id': r['id'],
            'title': r['title'] or 'Untitled',
            'user_name': r['user_name'] or 'Anonymous',
            'product_name': info['name'],
            'product_icon': info['icon'],
            'product_style': f"{info['color'].replace('bg-', 'bg-')}/20 text-{info['color'].split('-')[1]}-300",
            'score': int(score) if score else None,
            'score_color': score_color,
            'time_ago': time_ago(r['created_at']),
        })

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "stats": stats,
        "products": products,
        "top_users": top_users,
        "avg_scores": avg_scores,
        "daily_activity": json.dumps(daily_activity),
        "recent_user_recordings": recent_user_recordings,
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
    
    speeches = []
    
    # Try Supabase first for user recordings
    from speech_db import USE_SUPABASE, SUPABASE_AVAILABLE
    if USE_SUPABASE and SUPABASE_AVAILABLE:
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            import os
            conn_str = os.getenv(
                "DATABASE_URL",
                "postgresql://postgres.fkxuqyvcvxklzrxjmzsa:Y4ZLP97tHSTQn7Jz@aws-1-us-east-1.pooler.supabase.com:6543/postgres"
            )
            with psycopg2.connect(conn_str) as pg_conn:
                with pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Build query for Supabase
                    pg_query = """
                        SELECT s.id, s.title, s.speaker, s.year, s.source_url, s.source_file, s.duration_sec,
                               s.language, s.category, s.products, s.tags, s.notes, s.description, s.audio_hash,
                               s.created_at, s.updated_at, s.user_id, s.profile, s.project_id,
                               a.wpm, a.pitch_mean_hz, a.vocal_fry_ratio, a.lexical_diversity, 
                               a.quality_level, a.sentiment
                        FROM speeches s
                        LEFT JOIN analyses a ON a.speech_id = s.id
                        WHERE s.user_id IS NOT NULL
                    """
                    pg_params = []
                    if category:
                        pg_query += " AND s.category = %s"
                        pg_params.append(category)
                    if product:
                        pg_query += " AND s.products::text LIKE %s"
                        pg_params.append(f'%"{product}"%')
                    if q:
                        pg_query += " AND (LOWER(s.title) LIKE %s OR LOWER(s.speaker) LIKE %s OR LOWER(COALESCE(s.description, '')) LIKE %s)"
                        q_param = f"%{q.lower()}%"
                        pg_params.extend([q_param, q_param, q_param])
                    pg_query += " ORDER BY s.created_at DESC LIMIT 500"
                    
                    cur.execute(pg_query, pg_params)
                    rows = cur.fetchall()
                    
                    for row in rows:
                        d = dict(row)
                        # Handle products - might be list or JSON string
                        products = d.get("products")
                        if isinstance(products, str):
                            d["products"] = json.loads(products) if products else []
                        elif isinstance(products, list):
                            d["products"] = products
                        else:
                            d["products"] = []
                        d["metrics"] = {
                            k: d.pop(k) for k in ["wpm", "pitch_mean_hz", "vocal_fry_ratio", 
                                                   "lexical_diversity", "quality_level", "sentiment"]
                            if d.get(k) is not None
                        }
                        speeches.append(d)
        except Exception as e:
            logger.warning(f"Supabase query failed, falling back to SQLite: {e}")
    
    # Fall back to SQLite if Supabase failed or not enabled
    if not speeches:
        db = get_db()
        
        # Single optimized query with JOIN to get speeches + analysis in one shot
        query = """
            SELECT s.id, s.title, s.speaker, s.year, s.source_url, s.source_file, s.duration_sec,
                   s.language, s.category, s.products, s.tags, s.notes, s.description, s.audio_hash,
                   s.created_at, s.updated_at, s.user_id, s.profile, s.project_id,
                   a.wpm, a.pitch_mean_hz, a.vocal_fry_ratio, a.lexical_diversity, 
                   a.quality_level, a.sentiment
            FROM speeches s
            LEFT JOIN analyses a ON a.speech_id = s.id
            WHERE s.user_id IS NOT NULL
        """
        params = []
        if category:
            query += " AND s.category = ?"
            params.append(category)
        if product:
            query += " AND s.products LIKE ?"
            params.append(f'%"{product}"%')
        if q:
            query += " AND (LOWER(s.title) LIKE ? OR LOWER(s.speaker) LIKE ? OR LOWER(s.description) LIKE ?)"
            q_param = f"%{q.lower()}%"
            params.extend([q_param, q_param, q_param])
        query += " ORDER BY s.created_at DESC LIMIT 500"
        
        rows = db.conn.execute(query, params).fetchall()
        for row in rows:
            d = dict(row)
            d["products"] = json.loads(d["products"]) if d.get("products") else []
            d["metrics"] = {
                k: d.pop(k) for k in ["wpm", "pitch_mean_hz", "vocal_fry_ratio", 
                                       "lexical_diversity", "quality_level", "sentiment"]
                if d.get(k) is not None
            }
            speeches.append(d)

    # Get categories - use Supabase if available
    all_categories = []
    all_products = []
    
    if USE_SUPABASE and SUPABASE_AVAILABLE:
        try:
            with psycopg2.connect(conn_str) as pg_conn:
                with pg_conn.cursor() as cur:
                    cur.execute("SELECT DISTINCT category FROM speeches WHERE user_id IS NOT NULL AND category IS NOT NULL")
                    all_categories = sorted([r[0] for r in cur.fetchall()])
        except:
            pass
    
    if not all_categories:
        db = get_db()
        cat_rows = db.conn.execute(
            "SELECT DISTINCT category FROM speeches WHERE user_id IS NOT NULL AND category IS NOT NULL"
        ).fetchall()
        all_categories = sorted([r[0] for r in cat_rows])
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


@router.get("/interview", response_class=HTMLResponse)
async def interview_page(request: Request):
    """AI Interview Practice page."""
    return templates.TemplateResponse("interview.html", {
        "request": request,
    })


@router.get("/present", response_class=HTMLResponse)
async def present_page(request: Request):
    """Present Mode landing page."""
    return templates.TemplateResponse("present.html", {
        "request": request,
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
    
    # Time ranges
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    yesterday_start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    week_ago = (now - timedelta(days=7)).isoformat()
    week_start = (now - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    month_ago = (now - timedelta(days=30)).isoformat()
    two_months_ago = (now - timedelta(days=60)).isoformat()
    
    # DAU - Daily Active Users (users who made recordings today)
    dau = db.conn.execute("""
        SELECT COUNT(DISTINCT user_id) FROM speeches 
        WHERE user_id IS NOT NULL AND created_at >= ?
    """, (today_start,)).fetchone()[0]
    
    # DAU yesterday (for trend)
    dau_yesterday = db.conn.execute("""
        SELECT COUNT(DISTINCT user_id) FROM speeches 
        WHERE user_id IS NOT NULL AND created_at >= ? AND created_at < ?
    """, (yesterday_start, today_start)).fetchone()[0]
    
    # Active users (7 and 30 days)
    active_7d = db.conn.execute("""
        SELECT COUNT(DISTINCT user_id) FROM speeches 
        WHERE user_id IS NOT NULL AND created_at >= ?
    """, (week_ago,)).fetchone()[0]
    
    # MAU - Monthly Active Users
    mau = db.conn.execute("""
        SELECT COUNT(DISTINCT user_id) FROM speeches 
        WHERE user_id IS NOT NULL AND created_at >= ?
    """, (month_ago,)).fetchone()[0]
    
    # Previous month MAU (for MoM growth)
    mau_prev_month = db.conn.execute("""
        SELECT COUNT(DISTINCT user_id) FROM speeches 
        WHERE user_id IS NOT NULL AND created_at >= ? AND created_at < ?
    """, (two_months_ago, month_ago)).fetchone()[0]
    
    # Month-over-Month growth
    mom_growth = ((mau - mau_prev_month) / mau_prev_month * 100) if mau_prev_month > 0 else 0
    
    # Subscription stats by tier
    tier_counts = db.conn.execute("""
        SELECT COALESCE(tier, 'free') as tier, COUNT(*) as count
        FROM users
        GROUP BY tier
        ORDER BY count DESC
    """).fetchall()
    tier_stats = {row[0]: row[1] for row in tier_counts}
    # Count paid users (exclude free, none, admin, and beta testers)
    paid_users_row = db.conn.execute("""
        SELECT COUNT(*) FROM users 
        WHERE tier NOT IN ('free', 'none', 'admin') 
        AND tier IS NOT NULL 
        AND COALESCE(is_beta, 0) = 0
    """).fetchone()
    paid_users = paid_users_row[0] if paid_users_row else 0
    
    # New paid subscriptions (today, this week, this month)
    paid_today = db.conn.execute("""
        SELECT COUNT(*) FROM users 
        WHERE tier NOT IN ('free', 'none', 'admin') AND tier IS NOT NULL 
        AND COALESCE(is_beta, 0) = 0 AND created_at >= ?
    """, (today_start,)).fetchone()[0]
    
    paid_week = db.conn.execute("""
        SELECT COUNT(*) FROM users 
        WHERE tier NOT IN ('free', 'none', 'admin') AND tier IS NOT NULL 
        AND COALESCE(is_beta, 0) = 0 AND created_at >= ?
    """, (week_start,)).fetchone()[0]
    
    paid_month = db.conn.execute("""
        SELECT COUNT(*) FROM users 
        WHERE tier NOT IN ('free', 'none', 'admin') AND tier IS NOT NULL 
        AND COALESCE(is_beta, 0) = 0 AND created_at >= ?
    """, (month_ago,)).fetchone()[0]
    
    # DAU trend (last 14 days)
    dau_trend = db.conn.execute("""
        SELECT DATE(created_at) as date, COUNT(DISTINCT user_id) as dau
        FROM speeches
        WHERE user_id IS NOT NULL AND created_at >= ?
        GROUP BY DATE(created_at)
        ORDER BY date
    """, ((now - timedelta(days=14)).isoformat(),)).fetchall()
    
    # Recordings today and this week
    recordings_today = db.conn.execute("""
        SELECT COUNT(*) FROM speeches WHERE user_id IS NOT NULL AND created_at >= ?
    """, (today_start,)).fetchone()[0]
    
    recordings_week = db.conn.execute("""
        SELECT COUNT(*) FROM speeches WHERE user_id IS NOT NULL AND created_at >= ?
    """, (week_start,)).fetchone()[0]
    
    # Storage (audio files)
    audio_dir = Path("/home/melchior/speech3/audio")
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
        "dau_trend": [{"date": r[0], "dau": r[1]} for r in dau_trend],
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
    
    # System Health (CPU, Memory, GPU)
    system_health = {
        "cpu_percent": 0,
        "memory_percent": 0,
        "memory_used_gb": 0,
        "memory_total_gb": 0,
        "gpu_memory_used_gb": 0,
        "gpu_memory_total_gb": 0,
        "gpu_utilization": 0,
        "disk_percent": 0,
        "disk_used_gb": 0,
        "disk_total_gb": 0,
    }
    try:
        import psutil
        import subprocess
        
        # CPU and Memory (interval=None for instant value, no blocking)
        system_health["cpu_percent"] = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        system_health["memory_percent"] = mem.percent
        system_health["memory_used_gb"] = mem.used / (1024**3)
        system_health["memory_total_gb"] = mem.total / (1024**3)
        
        # Disk
        disk = psutil.disk_usage('/')
        system_health["disk_percent"] = disk.percent
        system_health["disk_used_gb"] = disk.used / (1024**3)
        system_health["disk_total_gb"] = disk.total / (1024**3)
        
        # GPU (nvidia-smi)
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                if len(parts) >= 3:
                    system_health["gpu_memory_used_gb"] = float(parts[0].strip()) / 1024
                    system_health["gpu_memory_total_gb"] = float(parts[1].strip()) / 1024
                    system_health["gpu_utilization"] = float(parts[2].strip())
        except:
            pass
    except Exception as e:
        pass
    
    # API Metrics
    api_metrics = {
        "requests": {"total": 0, "avg_latency_ms": 0, "p95_latency_ms": 0, "error_rate_pct": 0},
        "pipeline": {"total_processed": 0, "error_rate_pct": 0, "avg_duration_sec": 0},
        "queue": {"current_depth": 0, "max_depth": 20}
    }
    try:
        from api.metrics import metrics
        api_metrics = metrics.get_metrics()
    except Exception as e:
        pass
    
    # Retention metrics (D1, D7, D30)
    # D1: users who came back day after signup
    # D7: users who came back within 7 days of signup
    # D30: users who came back within 30 days of signup
    
    # Get users who signed up 1+ days ago and made recordings after signup date
    d1_eligible = db.conn.execute("""
        SELECT COUNT(DISTINCT u.id) FROM users u
        WHERE DATE(u.created_at) < DATE('now', '-1 day')
    """).fetchone()[0]
    
    d1_retained = db.conn.execute("""
        SELECT COUNT(DISTINCT u.id) FROM users u
        WHERE DATE(u.created_at) < DATE('now', '-1 day')
        AND EXISTS (
            SELECT 1 FROM speeches s 
            WHERE s.user_id = u.id 
            AND DATE(s.created_at) = DATE(u.created_at, '+1 day')
        )
    """).fetchone()[0]
    
    d7_eligible = db.conn.execute("""
        SELECT COUNT(DISTINCT u.id) FROM users u
        WHERE DATE(u.created_at) < DATE('now', '-7 day')
    """).fetchone()[0]
    
    d7_retained = db.conn.execute("""
        SELECT COUNT(DISTINCT u.id) FROM users u
        WHERE DATE(u.created_at) < DATE('now', '-7 day')
        AND EXISTS (
            SELECT 1 FROM speeches s 
            WHERE s.user_id = u.id 
            AND DATE(s.created_at) BETWEEN DATE(u.created_at, '+1 day') AND DATE(u.created_at, '+7 day')
        )
    """).fetchone()[0]
    
    d30_eligible = db.conn.execute("""
        SELECT COUNT(DISTINCT u.id) FROM users u
        WHERE DATE(u.created_at) < DATE('now', '-30 day')
    """).fetchone()[0]
    
    d30_retained = db.conn.execute("""
        SELECT COUNT(DISTINCT u.id) FROM users u
        WHERE DATE(u.created_at) < DATE('now', '-30 day')
        AND EXISTS (
            SELECT 1 FROM speeches s 
            WHERE s.user_id = u.id 
            AND DATE(s.created_at) BETWEEN DATE(u.created_at, '+1 day') AND DATE(u.created_at, '+30 day')
        )
    """).fetchone()[0]
    
    d1_rate = (d1_retained / d1_eligible * 100) if d1_eligible > 0 else 0
    d7_rate = (d7_retained / d7_eligible * 100) if d7_eligible > 0 else 0
    d30_rate = (d30_retained / d30_eligible * 100) if d30_eligible > 0 else 0
    
    # Churn metrics
    # Active last month = users who made recordings 30-60 days ago
    # Churned = those who didn't make recordings in the last 30 days
    active_last_month = db.conn.execute("""
        SELECT COUNT(DISTINCT user_id) FROM speeches
        WHERE user_id IS NOT NULL
        AND created_at >= DATE('now', '-60 day')
        AND created_at < DATE('now', '-30 day')
    """).fetchone()[0]
    
    churned_users = db.conn.execute("""
        SELECT COUNT(DISTINCT s1.user_id) FROM speeches s1
        WHERE s1.user_id IS NOT NULL
        AND s1.created_at >= DATE('now', '-60 day')
        AND s1.created_at < DATE('now', '-30 day')
        AND NOT EXISTS (
            SELECT 1 FROM speeches s2 
            WHERE s2.user_id = s1.user_id 
            AND s2.created_at >= DATE('now', '-30 day')
        )
    """).fetchone()[0]
    
    churn_rate = (churned_users / active_last_month * 100) if active_last_month > 0 else 0
    
    # Inactive users (no recordings in 30 days but account exists)
    inactive_users = db.conn.execute("""
        SELECT COUNT(*) FROM users u
        WHERE NOT EXISTS (
            SELECT 1 FROM speeches s 
            WHERE s.user_id = u.id 
            AND s.created_at >= DATE('now', '-30 day')
        )
    """).fetchone()[0]
    
    # Revenue metrics (MRR/ARR)
    TIER_PRICING = {
        "free": 0,
        "pro": 9.99,
        "enterprise": 49.99,
        "admin": 0,
        "none": 0,
    }
    
    # Calculate MRR from tier counts
    mrr = sum(TIER_PRICING.get(tier, 0) * count for tier, count in tier_stats.items())
    arr = mrr * 12
    
    # Revenue by tier breakdown
    revenue_by_tier = [
        {"tier": tier, "users": count, "price": TIER_PRICING.get(tier, 0), "mrr": TIER_PRICING.get(tier, 0) * count}
        for tier, count in tier_stats.items()
        if TIER_PRICING.get(tier, 0) > 0
    ]
    
    # Error cards from Trello
    error_cards = []
    error_count = 0
    try:
        import requests
        TRELLO_KEY = "94dbc1af84f0abce181fd9433f69d727"
        TRELLO_TOKEN = "ATTA5e7ac0ff9877573186a0a8979f4b5525c0f3c68efdf77514ad640ac653de267aAF589D6A"
        BACKLOG_LIST = "6980f79bb30707574404f57a"
        
        resp = requests.get(
            f"https://api.trello.com/1/lists/{BACKLOG_LIST}/cards",
            params={"key": TRELLO_KEY, "token": TRELLO_TOKEN},
            timeout=2  # Short timeout to avoid blocking
        )
        if resp.status_code == 200:
            cards = resp.json()
            error_cards = [c for c in cards if c.get("name", "").startswith("🐛")]
            error_count = len(error_cards)
    except Exception as e:
        pass
    
    # Cloud status (R2 + Supabase)
    cloud_status = {"status": "unknown", "supabase": {"connected": False}, "r2": {"connected": False}}
    try:
        from api.r2_storage import get_r2_storage
        import psycopg2
        from psycopg2.extras import RealDictCursor
        import os
        
        # Check Supabase
        supabase_url = os.getenv("DATABASE_URL")
        if supabase_url and os.getenv("USE_SUPABASE", "").lower() in ("true", "1", "yes"):
            try:
                conn = psycopg2.connect(supabase_url, cursor_factory=RealDictCursor, connect_timeout=5)
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) as c FROM users")
                cloud_status["supabase"] = {"connected": True, "users": cur.fetchone()["c"]}
                conn.close()
            except Exception as e:
                cloud_status["supabase"] = {"connected": False, "error": str(e)[:100]}
        
        # Check R2
        try:
            r2 = get_r2_storage()
            # Quick test - list with limit 1
            r2.client.list_objects_v2(Bucket=r2.bucket, MaxKeys=1)
            cloud_status["r2"] = {"connected": True, "bucket": r2.bucket}
        except Exception as e:
            cloud_status["r2"] = {"connected": False, "error": str(e)[:100]}
        
        cloud_status["status"] = "ok" if cloud_status["supabase"].get("connected") and cloud_status["r2"].get("connected") else "degraded"
    except Exception as e:
        cloud_status["status"] = "error"
        cloud_status["error"] = str(e)[:100]
    
    # Profile usage breakdown
    profile_usage = db.conn.execute("""
        SELECT COALESCE(profile, 'general') as profile, COUNT(*) as count
        FROM speeches
        WHERE user_id IS NOT NULL
        GROUP BY profile
        ORDER BY count DESC
    """).fetchall()
    
    # Average recordings per user
    avg_recordings_per_user = total_recordings / total_users if total_users > 0 else 0
    
    return templates.TemplateResponse("admin.html", {
        "request": request,
        "stats": {
            "total_users": total_users,
            "total_recordings": total_recordings,
            "dau": dau,
            "dau_yesterday": dau_yesterday,
            "mau": mau,
            "mau_prev_month": mau_prev_month,
            "mom_growth": mom_growth,
            "active_7d": active_7d,
            "active_30d": mau,  # Same as MAU
            "paid_users": paid_users,
            "paid_today": paid_today,
            "paid_week": paid_week,
            "paid_month": paid_month,
            "tier_stats": tier_stats,
            "recordings_today": recordings_today,
            "recordings_week": recordings_week,
            "storage_mb": storage_mb,
            "avg_score": avg_score,
            "survey_total": survey_total,
            "survey_completion_rate": survey_completion_rate,
            "avg_recordings_per_user": avg_recordings_per_user,
            "mrr": mrr,
            "arr": arr,
            "d1_rate": d1_rate,
            "d7_rate": d7_rate,
            "d30_rate": d30_rate,
            "d1_eligible": d1_eligible,
            "d7_eligible": d7_eligible,
            "d30_eligible": d30_eligible,
            "churn_rate": churn_rate,
            "churned_users": churned_users,
            "active_last_month": active_last_month,
            "inactive_users": inactive_users,
        },
        "revenue_by_tier": revenue_by_tier,
        "alerts": _generate_alerts(
            dau=dau,
            error_rate=api_metrics.get("requests", {}).get("error_rate_pct", 0),
            cpu=system_health.get("cpu_percent", 0),
            memory=system_health.get("memory_percent", 0),
            gpu=system_health.get("gpu_utilization", 0),
            disk=system_health.get("disk_percent", 0),
            r2_connected=cloud_status.get("r2", {}).get("connected", False),
            supabase_connected=cloud_status.get("supabase", {}).get("connected", False),
            churn_rate=churn_rate,
            error_cards=error_count,
        ),
        "cloud_status": cloud_status,
        "system_health": system_health,
        "api_metrics": api_metrics,
        "error_cards": error_cards[:10],  # Limit to 10 most recent
        "error_count": error_count,
        "profile_usage": [{"profile": p[0], "count": p[1]} for p in profile_usage],
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
    
    # Get onboarding survey
    import json
    survey = None
    survey_row = db.conn.execute(
        "SELECT * FROM user_surveys WHERE user_id = ?", (user_id,)
    ).fetchone()
    if survey_row:
        survey = dict(survey_row)
        # Parse focus_areas JSON
        if survey.get('focus_areas'):
            try:
                survey['focus_areas'] = json.loads(survey['focus_areas'])
            except:
                survey['focus_areas'] = []
    
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
        "survey": survey,
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


@router.post("/admin/users/{user_id}/toggle-beta")
async def admin_toggle_user_beta(request: Request, user_id: int):
    """Toggle a user's beta tester status (admin only)."""
    current_user = get_current_user(request)
    if not current_user or not is_admin_user(request):
        return HTMLResponse("<h1>Access Denied</h1>", status_code=403)
    
    db = get_db()
    row = db.conn.execute("SELECT is_beta FROM users WHERE id = ?", (user_id,)).fetchone()
    current_beta = row[0] if row and row[0] is not None else 0
    new_beta = 0 if current_beta else 1
    db.conn.execute("UPDATE users SET is_beta = ? WHERE id = ?", (new_beta, user_id))
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


# ============================================================================
# CSV Export Endpoints
# ============================================================================

from fastapi.responses import StreamingResponse
import csv
import io

@router.get("/admin/export/users", tags=["Admin Export"])
async def export_users_csv(request: Request):
    """Export all users to CSV."""
    if not is_admin_user(request):
        return HTMLResponse("<h1>Access Denied</h1>", status_code=403)
    
    db = get_db()
    rows = db.conn.execute("""
        SELECT u.id, u.name, u.email, u.tier, u.provider, u.created_at, u.last_login,
               COUNT(s.id) as recording_count,
               COALESCE(SUM(s.duration_sec), 0) as total_duration_sec
        FROM users u
        LEFT JOIN speeches s ON s.user_id = u.id
        GROUP BY u.id
        ORDER BY u.created_at DESC
    """).fetchall()
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Name', 'Email', 'Tier', 'Provider', 'Created', 'Last Login', 'Recordings', 'Total Duration (sec)'])
    for row in rows:
        writer.writerow(row)
    
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=speakfit_users.csv"}
    )


@router.get("/admin/export/recordings", tags=["Admin Export"])
async def export_recordings_csv(
    request: Request,
    start_date: str = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(None, description="End date (YYYY-MM-DD)"),
):
    """Export recordings to CSV with optional date range."""
    if not is_admin_user(request):
        return HTMLResponse("<h1>Access Denied</h1>", status_code=403)
    
    db = get_db()
    
    query = """
        SELECT s.id, s.title, s.speaker, s.profile, s.category, s.duration_sec,
               s.created_at, u.name as user_name, u.email as user_email,
               a.wpm, a.filler_ratio, a.lexical_diversity, a.quality_level
        FROM speeches s
        LEFT JOIN users u ON u.id = s.user_id
        LEFT JOIN analyses a ON a.speech_id = s.id
        WHERE s.user_id IS NOT NULL
    """
    params = []
    
    if start_date:
        query += " AND DATE(s.created_at) >= ?"
        params.append(start_date)
    if end_date:
        query += " AND DATE(s.created_at) <= ?"
        params.append(end_date)
    
    query += " ORDER BY s.created_at DESC"
    
    rows = db.conn.execute(query, params).fetchall()
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Title', 'Speaker', 'Profile', 'Category', 'Duration (sec)', 
                     'Created', 'User Name', 'User Email', 'WPM', 'Filler Ratio', 
                     'Lexical Diversity', 'Quality Level'])
    for row in rows:
        writer.writerow(row)
    
    output.seek(0)
    filename = f"speakfit_recordings_{start_date or 'all'}_{end_date or 'now'}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.get("/admin/export/metrics", tags=["Admin Export"])
async def export_metrics_csv(
    request: Request,
    days: int = Query(30, description="Number of days to export"),
):
    """Export daily metrics to CSV."""
    if not is_admin_user(request):
        return HTMLResponse("<h1>Access Denied</h1>", status_code=403)
    
    db = get_db()
    
    # Daily aggregates
    rows = db.conn.execute("""
        SELECT 
            DATE(s.created_at) as date,
            COUNT(DISTINCT s.user_id) as active_users,
            COUNT(*) as recordings,
            AVG(s.duration_sec) as avg_duration,
            COUNT(DISTINCT CASE WHEN u.tier NOT IN ('free', 'none', 'admin') AND COALESCE(u.is_beta, 0) = 0 THEN u.id END) as paid_users
        FROM speeches s
        LEFT JOIN users u ON u.id = s.user_id
        WHERE s.user_id IS NOT NULL
        AND s.created_at >= DATE('now', ? || ' days')
        GROUP BY DATE(s.created_at)
        ORDER BY date
    """, (f"-{days}",)).fetchall()
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Date', 'Active Users', 'Recordings', 'Avg Duration (sec)', 'Paid Users'])
    for row in rows:
        writer.writerow(row)
    
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=speakfit_metrics_{days}d.csv"}
    )
