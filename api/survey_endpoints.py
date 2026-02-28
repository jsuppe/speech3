"""
User Survey API Endpoints

Handles onboarding survey submission and retrieval.
Admin endpoints for viewing aggregate survey data.
"""

import json
import sqlite3
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.user_auth import get_current_user, get_admin_user

router = APIRouter(prefix="/v1", tags=["survey"])

DB_PATH = Path(__file__).parent.parent / 'speechscore.db'


# --- Models ---

class SurveyResponse(BaseModel):
    primary_goal: str  # public_speaking, pronunciation, professional, language_learning, acting, confidence, exploring
    skill_level: str   # beginner, intermediate, advanced, unsure
    focus_areas: List[str]  # pacing, clarity, vocal_variety, filler_words, breathing, pronunciation, confidence, structure
    context: str       # work_presentations, interviews, casual, video_calls, public_events, recording, academic, everyday
    time_commitment: str  # 5min, 15min, 30min, flexible
    native_language: Optional[str] = None


class SurveyData(BaseModel):
    user_id: int
    completed_at: str
    primary_goal: str
    skill_level: str
    focus_areas: List[str]
    context: str
    time_commitment: str
    native_language: Optional[str]


# --- Database Setup ---

def init_survey_table():
    """Create survey table if not exists"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_surveys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL UNIQUE,
            primary_goal TEXT NOT NULL,
            skill_level TEXT NOT NULL,
            focus_areas TEXT NOT NULL,
            context TEXT NOT NULL,
            time_commitment TEXT NOT NULL,
            native_language TEXT,
            completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()

# Initialize table on module load
init_survey_table()


# --- User Endpoints ---

@router.post("/users/me/survey")
async def submit_survey(survey: SurveyResponse, user = Depends(get_current_user)):
    """Submit onboarding survey (creates or updates)"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if survey already exists
    cursor.execute("SELECT id FROM user_surveys WHERE user_id = ?", (user['id'],))
    existing = cursor.fetchone()
    
    focus_areas_json = json.dumps(survey.focus_areas)
    
    if existing:
        # Update existing
        cursor.execute("""
            UPDATE user_surveys SET
                primary_goal = ?,
                skill_level = ?,
                focus_areas = ?,
                context = ?,
                time_commitment = ?,
                native_language = ?,
                completed_at = ?
            WHERE user_id = ?
        """, (
            survey.primary_goal,
            survey.skill_level,
            focus_areas_json,
            survey.context,
            survey.time_commitment,
            survey.native_language,
            datetime.utcnow().isoformat(),
            user['id']
        ))
    else:
        # Insert new
        cursor.execute("""
            INSERT INTO user_surveys 
            (user_id, primary_goal, skill_level, focus_areas, context, time_commitment, native_language)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            user['id'],
            survey.primary_goal,
            survey.skill_level,
            focus_areas_json,
            survey.context,
            survey.time_commitment,
            survey.native_language
        ))
    
    conn.commit()
    conn.close()
    
    return {"status": "saved", "user_id": user['id']}


@router.get("/users/me/survey")
async def get_my_survey(user = Depends(get_current_user)):
    """Get current user's survey (or null if not completed)"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM user_surveys WHERE user_id = ?
    """, (user['id'],))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return {"completed": False, "survey": None}
    
    return {
        "completed": True,
        "survey": {
            "primary_goal": row['primary_goal'],
            "skill_level": row['skill_level'],
            "focus_areas": json.loads(row['focus_areas']),
            "context": row['context'],
            "time_commitment": row['time_commitment'],
            "native_language": row['native_language'],
            "completed_at": row['completed_at']
        }
    }


# --- Admin Endpoints ---

@router.get("/admin/surveys")
async def get_all_surveys(
    limit: int = 100,
    offset: int = 0,
    admin = Depends(get_admin_user)
):
    """Get all survey responses (admin only)"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT s.*, u.name as user_name, u.email as user_email
        FROM user_surveys s
        JOIN users u ON s.user_id = u.id
        ORDER BY s.completed_at DESC
        LIMIT ? OFFSET ?
    """, (limit, offset))
    
    rows = cursor.fetchall()
    
    # Get total count
    cursor.execute("SELECT COUNT(*) FROM user_surveys")
    total = cursor.fetchone()[0]
    
    conn.close()
    
    surveys = []
    for row in rows:
        surveys.append({
            "user_id": row['user_id'],
            "user_name": row['user_name'],
            "user_email": row['user_email'],
            "primary_goal": row['primary_goal'],
            "skill_level": row['skill_level'],
            "focus_areas": json.loads(row['focus_areas']),
            "context": row['context'],
            "time_commitment": row['time_commitment'],
            "native_language": row['native_language'],
            "completed_at": row['completed_at']
        })
    
    return {
        "total": total,
        "surveys": surveys
    }


@router.get("/admin/surveys/user/{user_id}")
async def get_user_survey(user_id: int, admin = Depends(get_admin_user)):
    """Get a specific user's survey response (admin only)"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT s.*, u.name as user_name, u.email as user_email
        FROM user_surveys s
        JOIN users u ON s.user_id = u.id
        WHERE s.user_id = ?
    """, (user_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return {"completed": False, "survey": None}
    
    return {
        "completed": True,
        "survey": {
            "user_id": row['user_id'],
            "user_name": row['user_name'],
            "user_email": row['user_email'],
            "primary_goal": row['primary_goal'],
            "skill_level": row['skill_level'],
            "focus_areas": json.loads(row['focus_areas']),
            "context": row['context'],
            "time_commitment": row['time_commitment'],
            "native_language": row['native_language'],
            "completed_at": row['completed_at']
        }
    }


@router.get("/admin/surveys/stats")
async def get_survey_stats(admin = Depends(get_admin_user)):
    """Get aggregate survey statistics (admin only)"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    stats = {
        "total_responses": 0,
        "primary_goals": {},
        "skill_levels": {},
        "contexts": {},
        "time_commitments": {},
        "native_languages": {},
        "focus_areas": {}
    }
    
    # Total
    cursor.execute("SELECT COUNT(*) FROM user_surveys")
    stats["total_responses"] = cursor.fetchone()[0]
    
    # Primary goals
    cursor.execute("SELECT primary_goal, COUNT(*) as cnt FROM user_surveys GROUP BY primary_goal")
    for row in cursor.fetchall():
        stats["primary_goals"][row[0]] = row[1]
    
    # Skill levels
    cursor.execute("SELECT skill_level, COUNT(*) as cnt FROM user_surveys GROUP BY skill_level")
    for row in cursor.fetchall():
        stats["skill_levels"][row[0]] = row[1]
    
    # Contexts
    cursor.execute("SELECT context, COUNT(*) as cnt FROM user_surveys GROUP BY context")
    for row in cursor.fetchall():
        stats["contexts"][row[0]] = row[1]
    
    # Time commitments
    cursor.execute("SELECT time_commitment, COUNT(*) as cnt FROM user_surveys GROUP BY time_commitment")
    for row in cursor.fetchall():
        stats["time_commitments"][row[0]] = row[1]
    
    # Native languages
    cursor.execute("SELECT native_language, COUNT(*) as cnt FROM user_surveys WHERE native_language IS NOT NULL GROUP BY native_language")
    for row in cursor.fetchall():
        stats["native_languages"][row[0]] = row[1]
    
    # Focus areas (need to parse JSON)
    cursor.execute("SELECT focus_areas FROM user_surveys")
    for row in cursor.fetchall():
        areas = json.loads(row[0])
        for area in areas:
            stats["focus_areas"][area] = stats["focus_areas"].get(area, 0) + 1
    
    conn.close()
    
    return stats
