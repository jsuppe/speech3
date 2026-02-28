"""
Reference Speeches API — Great speeches for comparison.
"""

import os
import sqlite3
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import config
from .user_auth import flexible_auth

router = APIRouter(prefix="/v1/reference", tags=["reference"])


class ReferenceSpeech(BaseModel):
    id: int
    title: str
    speaker: str
    year: Optional[int]
    category: Optional[str]
    duration_sec: Optional[float]
    description: Optional[str]
    has_audio: bool


class ComparisonResult(BaseModel):
    user_speech_id: int
    reference_speech_id: int
    user_metrics: dict
    reference_metrics: dict
    comparison: dict


def get_db():
    """Get database connection."""
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@router.get("/speeches", response_model=list[ReferenceSpeech])
async def list_reference_speeches(auth: dict = Depends(flexible_auth)):
    """List all reference speeches available for comparison."""
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, title, speaker, year, category, duration_sec, description, source_file
            FROM speeches
            WHERE is_reference = 1
            ORDER BY speaker, title
        """)
        rows = cur.fetchall()
        
        return [
            ReferenceSpeech(
                id=row["id"],
                title=row["title"],
                speaker=row["speaker"],
                year=row["year"],
                category=row["category"],
                duration_sec=row["duration_sec"],
                description=row["description"],
                has_audio=bool(row["source_file"])
            )
            for row in rows
        ]
    finally:
        conn.close()


@router.get("/speeches/{speech_id}")
async def get_reference_speech(speech_id: int, auth: dict = Depends(flexible_auth)):
    """Get details for a reference speech."""
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, title, speaker, year, category, duration_sec, description, 
                   source_file, cached_score_json
            FROM speeches
            WHERE id = ? AND is_reference = 1
        """, (speech_id,))
        row = cur.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Reference speech not found")
        
        import json
        metrics = {}
        if row["cached_score_json"]:
            try:
                metrics = json.loads(row["cached_score_json"])
            except:
                pass
        
        return {
            "id": row["id"],
            "title": row["title"],
            "speaker": row["speaker"],
            "year": row["year"],
            "category": row["category"],
            "duration_sec": row["duration_sec"],
            "description": row["description"],
            "has_audio": bool(row["source_file"]),
            "metrics": metrics
        }
    finally:
        conn.close()


@router.get("/speeches/{speech_id}/audio")
async def get_reference_audio(speech_id: int):
    """Stream audio for a reference speech (public - no auth required for famous speeches)."""
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT source_file FROM speeches
            WHERE id = ? AND is_reference = 1
        """, (speech_id,))
        row = cur.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Reference speech not found")
        
        if not row["source_file"]:
            raise HTTPException(status_code=404, detail="No audio available for this speech")
        
        audio_path = row["source_file"]
        
        # Handle relative paths
        if not os.path.isabs(audio_path):
            audio_path = os.path.join(config.AUDIO_DIR, audio_path)
        
        if not os.path.exists(audio_path):
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        # Determine content type
        ext = os.path.splitext(audio_path)[1].lower()
        content_types = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".ogg": "audio/ogg",
            ".opus": "audio/opus",
            ".m4a": "audio/mp4",
            ".flac": "audio/flac",
        }
        content_type = content_types.get(ext, "audio/mpeg")
        
        return FileResponse(
            audio_path,
            media_type=content_type,
            filename=f"{speech_id}_{os.path.basename(audio_path)}"
        )
    finally:
        conn.close()


@router.get("/compare/{user_speech_id}/{reference_speech_id}")
async def compare_speeches(
    user_speech_id: int,
    reference_speech_id: int,
    auth: dict = Depends(flexible_auth)
):
    """Compare a user's speech against a reference speech."""
    conn = get_db()
    try:
        cur = conn.cursor()
        
        # Get user speech
        cur.execute("""
            SELECT id, title, speaker, cached_score_json, cached_score
            FROM speeches WHERE id = ?
        """, (user_speech_id,))
        user_row = cur.fetchone()
        
        if not user_row:
            raise HTTPException(status_code=404, detail="User speech not found")
        
        # Get reference speech
        cur.execute("""
            SELECT id, title, speaker, cached_score_json, cached_score
            FROM speeches WHERE id = ? AND is_reference = 1
        """, (reference_speech_id,))
        ref_row = cur.fetchone()
        
        if not ref_row:
            raise HTTPException(status_code=404, detail="Reference speech not found")
        
        import json
        
        user_metrics = {}
        if user_row["cached_score_json"]:
            try:
                user_metrics = json.loads(user_row["cached_score_json"])
            except:
                pass
        
        ref_metrics = {}
        if ref_row["cached_score_json"]:
            try:
                ref_metrics = json.loads(ref_row["cached_score_json"])
            except:
                pass
        
        # Calculate comparison deltas
        comparison = {
            "overall_score_delta": (user_row["cached_score"] or 0) - (ref_row["cached_score"] or 0),
            "metrics_comparison": {}
        }
        
        # Compare key oratory metrics
        oratory_keys = [
            "pitch_variation", "repetition_score", "pitch_range_semitones",
            "words_per_minute", "hnr_mean", "discourse_connectedness"
        ]
        
        for key in oratory_keys:
            user_val = user_metrics.get(key)
            ref_val = ref_metrics.get(key)
            if user_val is not None and ref_val is not None:
                comparison["metrics_comparison"][key] = {
                    "user": user_val,
                    "reference": ref_val,
                    "delta": user_val - ref_val,
                    "percent_of_reference": round(user_val / ref_val * 100, 1) if ref_val != 0 else None
                }
        
        return {
            "user_speech": {
                "id": user_row["id"],
                "title": user_row["title"],
                "speaker": user_row["speaker"],
                "score": user_row["cached_score"],
                "metrics": user_metrics
            },
            "reference_speech": {
                "id": ref_row["id"],
                "title": ref_row["title"],
                "speaker": ref_row["speaker"],
                "score": ref_row["cached_score"],
                "metrics": ref_metrics
            },
            "comparison": comparison
        }
    finally:
        conn.close()
