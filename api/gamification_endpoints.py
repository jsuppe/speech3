"""
Gamification API endpoints.

Provides streaks, XP, achievements, and level information.
"""

import sqlite3
from pathlib import Path
from typing import List, Optional
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from gamification import (
    get_gamification_status, get_all_achievements, check_achievements,
    award_xp, get_unnotified_achievements, mark_achievements_notified,
    DAILY_XP_GOAL, LEVEL_THRESHOLDS
)
from .user_auth import flexible_auth
from fastapi import HTTPException

router = APIRouter(prefix="/v1/gamification", tags=["gamification"])

DB_PATH = Path(__file__).parent.parent / 'speechscore.db'


class XPAwardRequest(BaseModel):
    activity_type: str
    bonus_xp: int = 0
    metadata: Optional[dict] = None


class AchievementNotifyRequest(BaseModel):
    achievement_ids: List[str]


@router.get("/status")
async def get_status(auth: dict = Depends(flexible_auth)):
    """Get complete gamification status for current user.
    
    Returns streak info, XP, level, and recent achievements.
    """
    user_id = auth.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="User authentication required for gamification")
    
    conn = sqlite3.connect(DB_PATH)
    try:
        status = get_gamification_status(conn, user_id)
        
        # Convert to dict for JSON response
        result = {
            'user_id': status.user_id,
            'streak': {
                'current': status.current_streak,
                'longest': status.longest_streak,
                'freezes_available': status.streak_freezes,
            },
            'xp': {
                'total': status.total_xp,
                'daily': status.daily_xp,
                'daily_goal': status.daily_goal,
                'daily_goal_met': status.daily_xp >= status.daily_goal,
            },
            'level': {
                'current': status.level,
                'progress': round(status.level_progress, 3),
                'xp_to_next': status.xp_to_next_level,
            },
            'achievements': {
                'earned': status.achievements_earned,
                'total': status.achievements_total,
                'recent': [asdict(a) for a in status.recent_achievements],
            },
        }
        
        return result
    finally:
        conn.close()


@router.get("/achievements")
async def list_achievements(auth: dict = Depends(flexible_auth)):
    """List all achievements with user's progress.
    
    Returns all available achievements grouped by category,
    showing which ones the user has earned.
    """
    user_id = auth.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="User authentication required for gamification")
    
    conn = sqlite3.connect(DB_PATH)
    try:
        achievements = get_all_achievements(conn, user_id)
        
        # Group by category
        by_category = {}
        for ach in achievements:
            cat = ach['category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append({
                'id': ach['id'],
                'name': ach['name'],
                'description': ach['description'],
                'icon': ach['icon'],
                'xp_reward': ach['xp_reward'],
                'earned': ach['earned_at'] is not None,
                'earned_at': ach['earned_at'],
            })
        
        return {
            'categories': by_category,
            'total': len(achievements),
            'earned': sum(1 for a in achievements if a['earned_at']),
        }
    finally:
        conn.close()


@router.post("/check")
async def check_and_award(auth: dict = Depends(flexible_auth)):
    """Check for any newly earned achievements.
    
    Call this after activities that might trigger achievements.
    Returns list of newly earned achievements.
    """
    user_id = auth.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="User authentication required")
    
    conn = sqlite3.connect(DB_PATH)
    try:
        newly_earned = check_achievements(conn, user_id)
        
        return {
            'newly_earned': [asdict(a) for a in newly_earned],
            'count': len(newly_earned),
        }
    finally:
        conn.close()


@router.get("/achievements/unnotified")
async def get_unnotified(auth: dict = Depends(flexible_auth)):
    """Get achievements that haven't been shown to the user yet.
    
    Use this for push notifications and in-app alerts.
    """
    user_id = auth.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="User authentication required")
    
    conn = sqlite3.connect(DB_PATH)
    try:
        unnotified = get_unnotified_achievements(conn, user_id)
        
        return {
            'achievements': [asdict(a) for a in unnotified],
            'count': len(unnotified),
        }
    finally:
        conn.close()


@router.post("/achievements/mark-notified")
async def mark_notified(request: AchievementNotifyRequest, auth: dict = Depends(flexible_auth)):
    """Mark achievements as notified (shown to user).
    
    Call after displaying achievement notification to user.
    """
    user_id = auth.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="User authentication required")
    
    conn = sqlite3.connect(DB_PATH)
    try:
        mark_achievements_notified(conn, user_id, request.achievement_ids)
        return {'status': 'ok', 'marked': len(request.achievement_ids)}
    finally:
        conn.close()


@router.get("/leaderboard")
async def get_leaderboard(period: str = "weekly"):
    """Get improvement leaderboard.
    
    Shows users ranked by XP earned in the period.
    Uses anonymous display names for privacy.
    
    Args:
        period: "daily", "weekly", or "monthly"
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Calculate date range
        from datetime import datetime, timedelta
        
        if period == "daily":
            since = datetime.utcnow().replace(hour=0, minute=0, second=0).isoformat()
        elif period == "weekly":
            today = datetime.utcnow()
            start = today - timedelta(days=today.weekday())
            since = start.replace(hour=0, minute=0, second=0).isoformat()
        else:  # monthly
            today = datetime.utcnow()
            since = today.replace(day=1, hour=0, minute=0, second=0).isoformat()
        
        cursor.execute("""
            SELECT 
                al.user_id,
                u.name,
                SUM(al.xp_earned) as period_xp,
                us.total_xp,
                us.current_streak
            FROM activity_log al
            JOIN users u ON al.user_id = u.id
            JOIN user_streaks us ON al.user_id = us.user_id
            WHERE al.created_at >= ?
            GROUP BY al.user_id
            ORDER BY period_xp DESC
            LIMIT 20
        """, (since,))
        
        leaderboard = []
        for i, row in enumerate(cursor.fetchall()):
            # Anonymize: show first letter + asterisks
            name = row[1] or "Anonymous"
            display_name = name[0] + "*" * (len(name) - 1) if len(name) > 1 else name
            
            leaderboard.append({
                'rank': i + 1,
                'display_name': display_name,
                'period_xp': row[2],
                'total_xp': row[3],
                'streak': row[4],
            })
        
        return {
            'period': period,
            'leaderboard': leaderboard,
        }
    finally:
        conn.close()


@router.get("/levels")
async def get_level_info():
    """Get level threshold information.
    
    Returns XP required for each level.
    """
    levels = []
    for i, threshold in enumerate(LEVEL_THRESHOLDS):
        next_threshold = LEVEL_THRESHOLDS[i + 1] if i + 1 < len(LEVEL_THRESHOLDS) else None
        levels.append({
            'level': i + 1,
            'xp_required': threshold,
            'xp_to_next': next_threshold - threshold if next_threshold else None,
        })
    
    return {
        'levels': levels,
        'daily_xp_goal': DAILY_XP_GOAL,
    }


@router.get("/weekly-recap")
async def get_weekly_recap(auth: dict = Depends(flexible_auth)):
    """Get weekly progress recap for the user.
    
    Returns stats comparing this week to last week.
    """
    user_id = auth.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="User authentication required")
    
    from datetime import datetime, timedelta
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        today = datetime.utcnow()
        week_start = today - timedelta(days=today.weekday())
        last_week_start = week_start - timedelta(days=7)
        
        week_start_str = week_start.replace(hour=0, minute=0, second=0).isoformat()
        last_week_start_str = last_week_start.replace(hour=0, minute=0, second=0).isoformat()
        
        # This week's stats
        cursor.execute("""
            SELECT 
                COALESCE(SUM(xp_earned), 0) as xp,
                COUNT(*) as activities,
                COUNT(DISTINCT date(created_at)) as active_days
            FROM activity_log
            WHERE user_id = ? AND created_at >= ?
        """, (user_id, week_start_str))
        this_week = cursor.fetchone()
        
        # Last week's stats
        cursor.execute("""
            SELECT 
                COALESCE(SUM(xp_earned), 0) as xp,
                COUNT(*) as activities,
                COUNT(DISTINCT date(created_at)) as active_days
            FROM activity_log
            WHERE user_id = ? AND created_at >= ? AND created_at < ?
        """, (user_id, last_week_start_str, week_start_str))
        last_week = cursor.fetchone()
        
        # Recordings this week
        cursor.execute("""
            SELECT COUNT(*) FROM speeches
            WHERE user_id = ? AND created_at >= ?
        """, (user_id, week_start_str))
        recordings_this_week = cursor.fetchone()[0]
        
        # Achievements earned this week
        cursor.execute("""
            SELECT a.name, a.icon FROM user_achievements ua
            JOIN achievements a ON ua.achievement_id = a.id
            WHERE ua.user_id = ? AND ua.earned_at >= ?
        """, (user_id, week_start_str))
        new_achievements = [{"name": r[0], "icon": r[1]} for r in cursor.fetchall()]
        
        # Calculate changes
        xp_change = this_week[0] - last_week[0] if last_week[0] > 0 else this_week[0]
        xp_change_pct = (xp_change / last_week[0] * 100) if last_week[0] > 0 else 0
        
        return {
            "this_week": {
                "xp_earned": this_week[0],
                "activities": this_week[1],
                "active_days": this_week[2],
                "recordings": recordings_this_week,
            },
            "last_week": {
                "xp_earned": last_week[0],
                "activities": last_week[1],
                "active_days": last_week[2],
            },
            "comparison": {
                "xp_change": xp_change,
                "xp_change_percent": round(xp_change_pct, 1),
                "trending": "up" if xp_change > 0 else "down" if xp_change < 0 else "same",
            },
            "new_achievements": new_achievements,
            "week_start": week_start_str[:10],
        }
    finally:
        conn.close()
