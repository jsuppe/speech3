"""
Metrics history endpoints for SpeakFit app.
Provides historical metric data for trend visualization.
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, Query

from . import config
from .user_auth import flexible_auth

router = APIRouter()


def get_db():
    """Get database connection."""
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@router.get("/history/{profile}")
async def get_metrics_history(
    profile: str,
    days: int = Query(30, ge=7, le=90),
    user_id: int = Depends(flexible_auth)
):
    """
    Get historical metrics for a specific profile (presentation, reading_fluency, etc.)
    Returns daily aggregates for charting.
    """
    conn = get_db()
    try:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Get all recordings for this profile in time range
        rows = conn.execute("""
            SELECT 
                s.id,
                date(s.created_at) as day,
                s.created_at,
                a.score,
                a.wpm,
                a.filler_ratio,
                a.pitch_mean,
                a.pitch_stddev,
                a.pause_ratio,
                a.intensity_mean,
                a.intensity_stddev
            FROM speeches s
            JOIN analyses a ON s.id = a.speech_id
            WHERE s.user_id = ? 
              AND s.profile = ?
              AND s.created_at >= ?
            ORDER BY s.created_at ASC
        """, (user_id, profile, cutoff)).fetchall()
        
        if not rows:
            return {"profile": profile, "days": days, "data": [], "aggregates": {}}
        
        # Build daily data points
        daily_data = {}
        all_metrics = {
            "pacing": [], "filler_rate": [], "confidence": [],
            "pitch_variation": [], "pause_quality": [], "energy": [],
            "wpm": [], "accuracy": []
        }
        
        for row in rows:
            day = row["day"]
            if day not in daily_data:
                daily_data[day] = {
                    "date": day,
                    "count": 0,
                    "pacing": [], "filler_rate": [], "confidence": [],
                    "pitch_variation": [], "pause_quality": [], "energy": [],
                    "wpm": [], "accuracy": []
                }
            
            d = daily_data[day]
            d["count"] += 1
            
            # Extract metrics based on available data
            wpm = row["wpm"]
            if wpm:
                d["wpm"].append(wpm)
                all_metrics["wpm"].append(wpm)
                # Pacing score: optimal range 130-150 WPM
                pacing = max(0, 100 - abs(wpm - 140) * 2) if wpm else None
                if pacing:
                    d["pacing"].append(pacing)
                    all_metrics["pacing"].append(pacing)
            
            filler = row["filler_ratio"]
            if filler is not None:
                d["filler_rate"].append(filler * 100)  # Convert to percentage
                all_metrics["filler_rate"].append(filler * 100)
            
            # Confidence based on overall score (placeholder - would use hedging analysis)
            score = row["score"]
            if score:
                d["confidence"].append(score)
                all_metrics["confidence"].append(score)
                d["accuracy"].append(score)
                all_metrics["accuracy"].append(score)
            
            # Pitch variation (coefficient of variation)
            pitch_mean = row["pitch_mean"]
            pitch_std = row["pitch_stddev"]
            if pitch_mean and pitch_std and pitch_mean > 0:
                pitch_cv = (pitch_std / pitch_mean) * 100
                d["pitch_variation"].append(pitch_cv)
                all_metrics["pitch_variation"].append(pitch_cv)
            
            # Pause quality
            pause_ratio = row["pause_ratio"]
            if pause_ratio is not None:
                d["pause_quality"].append(pause_ratio * 100)
                all_metrics["pause_quality"].append(pause_ratio * 100)
            
            # Energy (intensity variation)
            int_mean = row["intensity_mean"]
            int_std = row["intensity_stddev"]
            if int_mean and int_std:
                # Normalize to 0-100 scale
                energy = min(100, max(0, int_mean / 0.5 * 100))
                d["energy"].append(energy)
                all_metrics["energy"].append(energy)
        
        # Average daily metrics
        result_data = []
        for day in sorted(daily_data.keys()):
            d = daily_data[day]
            entry = {
                "date": day,
                "count": d["count"],
            }
            for metric in ["pacing", "filler_rate", "confidence", "pitch_variation", 
                          "pause_quality", "energy", "wpm", "accuracy"]:
                vals = d[metric]
                entry[metric] = round(sum(vals) / len(vals), 1) if vals else None
            result_data.append(entry)
        
        # Overall aggregates
        aggregates = {}
        for metric, vals in all_metrics.items():
            if vals:
                aggregates[metric] = {
                    "avg": round(sum(vals) / len(vals), 1),
                    "min": round(min(vals), 1),
                    "max": round(max(vals), 1),
                    "trend": _calc_trend(vals),
                }
        
        return {
            "profile": profile,
            "days": days,
            "total_recordings": len(rows),
            "data": result_data,
            "aggregates": aggregates
        }
    finally:
        conn.close()


def _calc_trend(values: list) -> str:
    """Calculate simple trend direction."""
    if len(values) < 2:
        return "stable"
    
    # Compare first half to second half
    mid = len(values) // 2
    first_half = sum(values[:mid]) / mid if mid > 0 else 0
    second_half = sum(values[mid:]) / (len(values) - mid) if len(values) > mid else 0
    
    diff = second_half - first_half
    threshold = first_half * 0.05  # 5% change threshold
    
    if diff > threshold:
        return "improving"
    elif diff < -threshold:
        return "declining"
    return "stable"


@router.get("/stats/{profile}")
async def get_profile_stats(
    profile: str,
    user_id: int = Depends(flexible_auth)
):
    """
    Get current statistics for a profile (for home screen display).
    """
    conn = get_db()
    try:
        # Get recent recordings (last 30 days)
        cutoff = (datetime.now() - timedelta(days=30)).isoformat()
        
        rows = conn.execute("""
            SELECT 
                a.score,
                a.wpm,
                a.filler_ratio,
                a.pitch_mean,
                a.pitch_stddev,
                a.pause_ratio,
                a.intensity_mean,
                a.intensity_stddev
            FROM speeches s
            JOIN analyses a ON s.id = a.speech_id
            WHERE s.user_id = ? 
              AND s.profile = ?
              AND s.created_at >= ?
        """, (user_id, profile, cutoff)).fetchall()
        
        if not rows:
            return {
                "profile": profile,
                "total_recordings": 0,
                "avg_pacing_score": None,
                "avg_filler_rate": None,
                "avg_confidence": None,
                "pitch_cv": None,
                "pause_ratio": None,
                "energy_score": None,
                "avg_wpm": None,
                "avg_accuracy": None,
            }
        
        # Aggregate metrics
        metrics = {
            "pacing": [], "filler": [], "confidence": [],
            "pitch_cv": [], "pause": [], "energy": [],
            "wpm": [], "accuracy": []
        }
        
        for row in rows:
            if row["wpm"]:
                metrics["wpm"].append(row["wpm"])
                pacing = max(0, 100 - abs(row["wpm"] - 140) * 2)
                metrics["pacing"].append(pacing)
            
            if row["filler_ratio"] is not None:
                metrics["filler"].append(row["filler_ratio"] * 100)
            
            if row["score"]:
                metrics["confidence"].append(row["score"])
                metrics["accuracy"].append(row["score"])
            
            if row["pitch_mean"] and row["pitch_stddev"] and row["pitch_mean"] > 0:
                metrics["pitch_cv"].append((row["pitch_stddev"] / row["pitch_mean"]) * 100)
            
            if row["pause_ratio"] is not None:
                metrics["pause"].append(row["pause_ratio"] * 100)
            
            if row["intensity_mean"]:
                metrics["energy"].append(min(100, row["intensity_mean"] / 0.5 * 100))
        
        def avg(lst):
            return round(sum(lst) / len(lst), 1) if lst else None
        
        # Calculate streak
        streak = _calc_practice_streak(conn, user_id, profile)
        
        return {
            "profile": profile,
            "total_recordings": len(rows),
            "practice_streak": streak,
            "avg_pacing_score": avg(metrics["pacing"]),
            "avg_filler_rate": avg(metrics["filler"]),
            "avg_confidence": avg(metrics["confidence"]),
            "pitch_cv": avg(metrics["pitch_cv"]),
            "pause_ratio": avg(metrics["pause"]),
            "energy_score": avg(metrics["energy"]),
            "avg_wpm": avg(metrics["wpm"]),
            "avg_accuracy": avg(metrics["accuracy"]),
        }
    finally:
        conn.close()


def _calc_practice_streak(conn, user_id: int, profile: str) -> int:
    """Calculate consecutive days of practice."""
    rows = conn.execute("""
        SELECT DISTINCT date(created_at) as day
        FROM speeches
        WHERE user_id = ? AND profile = ?
        ORDER BY day DESC
        LIMIT 90
    """, (user_id, profile)).fetchall()
    
    if not rows:
        return 0
    
    streak = 0
    today = datetime.now().date()
    expected = today
    
    for row in rows:
        day = datetime.strptime(row["day"], "%Y-%m-%d").date()
        if day == expected:
            streak += 1
            expected -= timedelta(days=1)
        elif day == expected - timedelta(days=1):
            # Allow one day gap (yesterday counts for today)
            streak += 1
            expected = day - timedelta(days=1)
        else:
            break
    
    return streak
