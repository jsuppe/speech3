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

logger = logging.getLogger("speechscore.frontend")

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
async def speech_detail(request: Request, speech_id: int):
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

    return templates.TemplateResponse("speech_detail.html", {
        "request": request,
        "speech": speech,
        "analysis": analysis,
        "transcription": transcription,
        "has_audio": audio is not None,
        "audio_size": audio["size_bytes"] if audio else 0,
        "has_spectrogram": spectrogram is not None,
        "full_result": json.dumps(full_result, indent=2) if full_result else "{}",
    })


@router.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Upload page for new audio analysis."""
    return templates.TemplateResponse("upload.html", {
        "request": request,
    })


@router.get("/compare", response_class=HTMLResponse)
async def compare_page(
    request: Request,
    ids: str = Query(None),
):
    """Side-by-side speech comparison."""
    db = get_db()
    speeches = db.list_speeches(limit=500)

    compared = []
    if ids:
        id_list = [int(x.strip()) for x in ids.split(",") if x.strip().isdigit()]
        for sid in id_list[:4]:  # max 4 comparisons
            speech = db.get_speech(sid)
            if speech:
                analysis = db.get_analysis(sid)
                transcription = db.get_transcription(sid)
                compared.append({
                    "speech": speech,
                    "analysis": analysis,
                    "transcription": transcription,
                })

    return templates.TemplateResponse("compare.html", {
        "request": request,
        "speeches": speeches,
        "compared": compared,
        "current_ids": ids or "",
    })
