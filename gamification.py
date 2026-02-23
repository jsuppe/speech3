"""
Gamification system for SpeakFit.

Handles streaks, XP, achievements, and level progression.
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

DB_PATH = Path(__file__).parent / 'speechscore.db'

# XP values for activities
XP_VALUES = {
    'recording': 25,
    'pronunciation_practice': 15,
    'coach_chat': 5,
    'perfect_score': 50,  # bonus for 90%+ score
    'improvement': 10,    # bonus for beating personal best
}

# Level thresholds (XP required for each level)
LEVEL_THRESHOLDS = [
    0, 50, 150, 300, 500, 750, 1000, 1500, 2000, 3000,  # 1-10
    4000, 5000, 6500, 8000, 10000, 12500, 15000, 18000, 21000, 25000,  # 11-20
]

DAILY_XP_GOAL = 50


@dataclass
class Achievement:
    id: str
    name: str
    description: str
    icon: str
    category: str
    xp_reward: int
    requirement_type: str
    requirement_value: int
    earned_at: Optional[str] = None


@dataclass
class GamificationStatus:
    user_id: int
    current_streak: int
    longest_streak: int
    total_xp: int
    daily_xp: int
    daily_goal: int
    level: int
    level_progress: float  # 0.0 to 1.0
    xp_to_next_level: int
    streak_freezes: int
    achievements_earned: int
    achievements_total: int
    recent_achievements: List[Achievement]


def get_level(total_xp: int) -> tuple[int, float, int]:
    """Calculate level from total XP. Returns (level, progress, xp_to_next)."""
    level = 1
    for i, threshold in enumerate(LEVEL_THRESHOLDS):
        if total_xp >= threshold:
            level = i + 1
        else:
            break
    
    if level >= len(LEVEL_THRESHOLDS):
        return level, 1.0, 0
    
    current_threshold = LEVEL_THRESHOLDS[level - 1]
    next_threshold = LEVEL_THRESHOLDS[level] if level < len(LEVEL_THRESHOLDS) else current_threshold
    
    xp_in_level = total_xp - current_threshold
    xp_needed = next_threshold - current_threshold
    progress = xp_in_level / xp_needed if xp_needed > 0 else 1.0
    xp_to_next = next_threshold - total_xp
    
    return level, progress, xp_to_next


def get_user_streak(conn: sqlite3.Connection, user_id: int) -> dict:
    """Get or create user streak record."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_streaks WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    
    if not row:
        cursor.execute("""
            INSERT INTO user_streaks (user_id, current_streak, longest_streak, total_xp, daily_xp, streak_freezes)
            VALUES (?, 0, 0, 0, 0, 0)
        """, (user_id,))
        conn.commit()
        return {
            'current_streak': 0,
            'longest_streak': 0,
            'last_activity_date': None,
            'total_xp': 0,
            'daily_xp': 0,
            'daily_xp_date': None,
            'streak_freezes': 0,
        }
    
    cols = [desc[0] for desc in cursor.description]
    return dict(zip(cols, row))


def update_streak(conn: sqlite3.Connection, user_id: int) -> dict:
    """Update user's streak based on activity. Call after any activity."""
    cursor = conn.cursor()
    streak_data = get_user_streak(conn, user_id)
    
    today = datetime.utcnow().strftime('%Y-%m-%d')
    last_date = streak_data.get('last_activity_date')
    
    current_streak = streak_data['current_streak']
    longest_streak = streak_data['longest_streak']
    
    if last_date == today:
        # Already active today, no streak change
        pass
    elif last_date:
        last = datetime.strptime(last_date, '%Y-%m-%d')
        today_dt = datetime.strptime(today, '%Y-%m-%d')
        days_diff = (today_dt - last).days
        
        if days_diff == 1:
            # Consecutive day, increment streak
            current_streak += 1
        elif days_diff > 1:
            # Streak broken
            current_streak = 1
    else:
        # First activity ever
        current_streak = 1
    
    if current_streak > longest_streak:
        longest_streak = current_streak
    
    # Reset daily XP if new day
    daily_xp_date = streak_data.get('daily_xp_date')
    daily_xp = streak_data['daily_xp'] if daily_xp_date == today else 0
    
    cursor.execute("""
        UPDATE user_streaks 
        SET current_streak = ?, longest_streak = ?, last_activity_date = ?,
            daily_xp = ?, daily_xp_date = ?, updated_at = datetime('now')
        WHERE user_id = ?
    """, (current_streak, longest_streak, today, daily_xp, today, user_id))
    conn.commit()
    
    return {
        'current_streak': current_streak,
        'longest_streak': longest_streak,
        'streak_increased': last_date != today,
    }


def award_xp(conn: sqlite3.Connection, user_id: int, activity_type: str, 
             bonus_xp: int = 0, metadata: dict = None) -> dict:
    """Award XP for an activity. Returns XP earned and any level-up info."""
    cursor = conn.cursor()
    streak_data = get_user_streak(conn, user_id)
    
    base_xp = XP_VALUES.get(activity_type, 10)
    total_earned = base_xp + bonus_xp
    
    old_xp = streak_data['total_xp']
    new_xp = old_xp + total_earned
    
    today = datetime.utcnow().strftime('%Y-%m-%d')
    daily_xp_date = streak_data.get('daily_xp_date')
    daily_xp = streak_data['daily_xp'] if daily_xp_date == today else 0
    new_daily_xp = daily_xp + total_earned
    
    # Check for level up
    old_level, _, _ = get_level(old_xp)
    new_level, progress, xp_to_next = get_level(new_xp)
    leveled_up = new_level > old_level
    
    # Update streak data
    cursor.execute("""
        UPDATE user_streaks 
        SET total_xp = ?, daily_xp = ?, daily_xp_date = ?, updated_at = datetime('now')
        WHERE user_id = ?
    """, (new_xp, new_daily_xp, today, user_id))
    
    # Log activity
    import json
    cursor.execute("""
        INSERT INTO activity_log (user_id, activity_type, xp_earned, metadata_json)
        VALUES (?, ?, ?, ?)
    """, (user_id, activity_type, total_earned, json.dumps(metadata) if metadata else None))
    
    conn.commit()
    
    # Update streak
    update_streak(conn, user_id)
    
    return {
        'xp_earned': total_earned,
        'total_xp': new_xp,
        'daily_xp': new_daily_xp,
        'daily_goal': DAILY_XP_GOAL,
        'daily_goal_met': new_daily_xp >= DAILY_XP_GOAL,
        'level': new_level,
        'level_progress': progress,
        'xp_to_next_level': xp_to_next,
        'leveled_up': leveled_up,
        'old_level': old_level if leveled_up else None,
    }


def check_achievements(conn: sqlite3.Connection, user_id: int) -> List[Achievement]:
    """Check and award any newly earned achievements."""
    cursor = conn.cursor()
    newly_earned = []
    
    # Get user stats
    cursor.execute("SELECT COUNT(*) FROM speeches WHERE user_id = ?", (user_id,))
    recording_count = cursor.fetchone()[0]
    
    streak_data = get_user_streak(conn, user_id)
    current_streak = streak_data['current_streak']
    longest_streak = streak_data['longest_streak']
    
    # Get coach chat count
    cursor.execute("SELECT COUNT(*) FROM coach_messages WHERE user_id = ? AND role = 'user'", (user_id,))
    coach_chats = cursor.fetchone()[0]
    
    # Get already earned achievements
    cursor.execute("SELECT achievement_id FROM user_achievements WHERE user_id = ?", (user_id,))
    earned_ids = {row[0] for row in cursor.fetchall()}
    
    # Get all achievements
    cursor.execute("SELECT * FROM achievements")
    cols = [desc[0] for desc in cursor.description]
    achievements = [dict(zip(cols, row)) for row in cursor.fetchall()]
    
    for ach in achievements:
        if ach['id'] in earned_ids:
            continue
        
        earned = False
        req_type = ach['requirement_type']
        req_value = ach['requirement_value']
        
        # Check each requirement type
        if req_type == 'recordings' and recording_count >= req_value:
            earned = True
        elif req_type == 'streak' and longest_streak >= req_value:
            earned = True
        elif req_type == 'coach_chats' and coach_chats >= req_value:
            earned = True
        elif req_type == 'time_early':
            hour = datetime.utcnow().hour
            if hour < 7:
                earned = True
        elif req_type == 'time_late':
            hour = datetime.utcnow().hour
            if hour >= 22:
                earned = True
        # Add more requirement checks as needed
        
        if earned:
            cursor.execute("""
                INSERT INTO user_achievements (user_id, achievement_id)
                VALUES (?, ?)
            """, (user_id, ach['id']))
            
            # Award XP for achievement
            award_xp(conn, user_id, 'achievement', bonus_xp=ach['xp_reward'], 
                    metadata={'achievement_id': ach['id']})
            
            newly_earned.append(Achievement(
                id=ach['id'],
                name=ach['name'],
                description=ach['description'],
                icon=ach['icon'],
                category=ach['category'],
                xp_reward=ach['xp_reward'],
                requirement_type=ach['requirement_type'],
                requirement_value=ach['requirement_value'],
                earned_at=datetime.utcnow().isoformat(),
            ))
    
    conn.commit()
    return newly_earned


def get_gamification_status(conn: sqlite3.Connection, user_id: int) -> GamificationStatus:
    """Get complete gamification status for a user."""
    cursor = conn.cursor()
    streak_data = get_user_streak(conn, user_id)
    
    level, progress, xp_to_next = get_level(streak_data['total_xp'])
    
    # Get achievement counts
    cursor.execute("SELECT COUNT(*) FROM achievements")
    total_achievements = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM user_achievements WHERE user_id = ?", (user_id,))
    earned_achievements = cursor.fetchone()[0]
    
    # Get recent achievements
    cursor.execute("""
        SELECT a.*, ua.earned_at 
        FROM user_achievements ua
        JOIN achievements a ON ua.achievement_id = a.id
        WHERE ua.user_id = ?
        ORDER BY ua.earned_at DESC
        LIMIT 5
    """, (user_id,))
    cols = [desc[0] for desc in cursor.description]
    recent = []
    for row in cursor.fetchall():
        d = dict(zip(cols, row))
        recent.append(Achievement(
            id=d['id'],
            name=d['name'],
            description=d['description'],
            icon=d['icon'],
            category=d['category'],
            xp_reward=d['xp_reward'],
            requirement_type=d['requirement_type'],
            requirement_value=d['requirement_value'],
            earned_at=d['earned_at'],
        ))
    
    today = datetime.utcnow().strftime('%Y-%m-%d')
    daily_xp = streak_data['daily_xp'] if streak_data.get('daily_xp_date') == today else 0
    
    return GamificationStatus(
        user_id=user_id,
        current_streak=streak_data['current_streak'],
        longest_streak=streak_data['longest_streak'],
        total_xp=streak_data['total_xp'],
        daily_xp=daily_xp,
        daily_goal=DAILY_XP_GOAL,
        level=level,
        level_progress=progress,
        xp_to_next_level=xp_to_next,
        streak_freezes=streak_data['streak_freezes'],
        achievements_earned=earned_achievements,
        achievements_total=total_achievements,
        recent_achievements=recent,
    )


def get_all_achievements(conn: sqlite3.Connection, user_id: int) -> List[dict]:
    """Get all achievements with user's progress/earned status."""
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT a.*, ua.earned_at
        FROM achievements a
        LEFT JOIN user_achievements ua ON a.id = ua.achievement_id AND ua.user_id = ?
        ORDER BY a.category, a.requirement_value
    """, (user_id,))
    
    cols = [desc[0] for desc in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def get_unnotified_achievements(conn: sqlite3.Connection, user_id: int) -> List[Achievement]:
    """Get achievements earned but not yet notified."""
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT a.*, ua.earned_at
        FROM user_achievements ua
        JOIN achievements a ON ua.achievement_id = a.id
        WHERE ua.user_id = ? AND ua.notified_at IS NULL
    """, (user_id,))
    
    cols = [desc[0] for desc in cursor.description]
    achievements = []
    for row in cursor.fetchall():
        d = dict(zip(cols, row))
        achievements.append(Achievement(
            id=d['id'],
            name=d['name'],
            description=d['description'],
            icon=d['icon'],
            category=d['category'],
            xp_reward=d['xp_reward'],
            requirement_type=d['requirement_type'],
            requirement_value=d['requirement_value'],
            earned_at=d['earned_at'],
        ))
    
    return achievements


def mark_achievements_notified(conn: sqlite3.Connection, user_id: int, achievement_ids: List[str]):
    """Mark achievements as notified."""
    cursor = conn.cursor()
    cursor.execute(f"""
        UPDATE user_achievements 
        SET notified_at = datetime('now')
        WHERE user_id = ? AND achievement_id IN ({','.join('?' * len(achievement_ids))})
    """, [user_id] + achievement_ids)
    conn.commit()
