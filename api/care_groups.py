"""
Care Groups API - Multi-profile support for Memory app.

Enables caregivers to monitor multiple people under one account.
"""

import logging
import secrets
import string
from typing import Optional, List
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from .user_auth import flexible_auth
from .supabase_client import get_connection

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/groups", tags=["Care Groups"])


# --- Pydantic Models ---

class GroupCreate(BaseModel):
    name: str = Field(default="My Family", max_length=100)
    type: str = Field(default="family")  # family, clinic, facility

class GroupUpdate(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    timezone: Optional[str] = None
    settings: Optional[dict] = None

class ProfileCreate(BaseModel):
    name: str = Field(..., max_length=100)
    avatar_emoji: str = Field(default="👤", max_length=10)
    avatar_color: str = Field(default="#14B8A6", max_length=20)
    date_of_birth: Optional[str] = None
    diagnosis: Optional[str] = None
    notes: Optional[str] = None
    physician_name: Optional[str] = None
    physician_email: Optional[str] = None

class ProfileUpdate(BaseModel):
    name: Optional[str] = None
    avatar_emoji: Optional[str] = None
    avatar_color: Optional[str] = None
    date_of_birth: Optional[str] = None
    diagnosis: Optional[str] = None
    notes: Optional[str] = None
    physician_name: Optional[str] = None
    physician_email: Optional[str] = None
    settings: Optional[dict] = None

class InviteCreate(BaseModel):
    role: str = Field(default="caregiver")  # admin, caregiver, viewer
    email: Optional[str] = None  # optional: restrict to specific email

class JoinGroup(BaseModel):
    invite_code: str


# --- Helper Functions ---

def generate_invite_code(length: int = 8) -> str:
    """Generate a random alphanumeric invite code."""
    chars = string.ascii_uppercase + string.digits
    # Remove ambiguous characters
    chars = chars.replace('O', '').replace('0', '').replace('I', '').replace('1', '')
    return ''.join(secrets.choice(chars) for _ in range(length))


def get_user_role_in_group(user_id: int, group_id: str) -> Optional[str]:
    """Get user's role in a group, or None if not a member."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT role FROM group_members WHERE user_id = %s AND group_id = %s",
        (user_id, group_id)
    )
    row = cur.fetchone()
    conn.close()
    return row['role'] if row else None


def require_group_role(user_id: int, group_id: str, allowed_roles: List[str]):
    """Raise 403 if user doesn't have required role."""
    role = get_user_role_in_group(user_id, group_id)
    if role is None:
        raise HTTPException(status_code=404, detail="Group not found")
    if role not in allowed_roles:
        raise HTTPException(status_code=403, detail=f"Requires role: {', '.join(allowed_roles)}")
    return role


# --- Group Endpoints ---

@router.get("")
async def list_groups(auth: dict = Depends(flexible_auth)):
    """List all groups the user belongs to."""
    user_id = auth.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('''
        SELECT g.*, m.role as my_role,
               (SELECT COUNT(*) FROM care_profiles WHERE group_id = g.id) as profile_count,
               (SELECT COUNT(*) FROM group_members WHERE group_id = g.id) as member_count
        FROM care_groups g
        JOIN group_members m ON m.group_id = g.id
        WHERE m.user_id = %s
        ORDER BY g.created_at DESC
    ''', (user_id,))
    groups = cur.fetchall()
    conn.close()
    
    return {"groups": [dict(g) for g in groups]}


@router.post("")
async def create_group(data: GroupCreate, auth: dict = Depends(flexible_auth)):
    """Create a new care group."""
    user_id = auth.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    if data.type not in ('family', 'clinic', 'facility'):
        raise HTTPException(status_code=400, detail="Type must be: family, clinic, or facility")
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Create the group
    cur.execute('''
        INSERT INTO care_groups (owner_id, name, type)
        VALUES (%s, %s, %s)
        RETURNING id, name, type, timezone, settings, created_at
    ''', (user_id, data.name, data.type))
    group = dict(cur.fetchone())
    
    # Add creator as admin member
    cur.execute('''
        INSERT INTO group_members (group_id, user_id, role)
        VALUES (%s, %s, 'admin')
    ''', (group['id'], user_id))
    
    conn.commit()
    conn.close()
    
    logger.info(f"User {user_id} created group {group['id']}: {data.name}")
    return {"group": group, "message": "Group created successfully"}


@router.get("/{group_id}")
async def get_group(group_id: str, auth: dict = Depends(flexible_auth)):
    """Get group details."""
    user_id = auth.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    require_group_role(user_id, group_id, ['admin', 'caregiver', 'viewer'])
    
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('''
        SELECT g.*, 
               (SELECT COUNT(*) FROM care_profiles WHERE group_id = g.id) as profile_count,
               (SELECT COUNT(*) FROM group_members WHERE group_id = g.id) as member_count
        FROM care_groups g
        WHERE g.id = %s
    ''', (group_id,))
    group = cur.fetchone()
    conn.close()
    
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    
    return {"group": dict(group)}


@router.patch("/{group_id}")
async def update_group(group_id: str, data: GroupUpdate, auth: dict = Depends(flexible_auth)):
    """Update group settings (admin only)."""
    user_id = auth.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    require_group_role(user_id, group_id, ['admin'])
    
    updates = []
    values = []
    if data.name is not None:
        updates.append("name = %s")
        values.append(data.name)
    if data.type is not None:
        if data.type not in ('family', 'clinic', 'facility'):
            raise HTTPException(status_code=400, detail="Invalid type")
        updates.append("type = %s")
        values.append(data.type)
    if data.timezone is not None:
        updates.append("timezone = %s")
        values.append(data.timezone)
    if data.settings is not None:
        updates.append("settings = %s")
        values.append(data.settings)
    
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    
    updates.append("updated_at = NOW()")
    values.append(group_id)
    
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(f'''
        UPDATE care_groups SET {", ".join(updates)}
        WHERE id = %s
        RETURNING *
    ''', values)
    group = cur.fetchone()
    conn.commit()
    conn.close()
    
    return {"group": dict(group)}


@router.delete("/{group_id}")
async def delete_group(group_id: str, auth: dict = Depends(flexible_auth)):
    """Delete a group (admin only). All profiles and their recordings will be unlinked."""
    user_id = auth.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    require_group_role(user_id, group_id, ['admin'])
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Unlink recordings (don't delete, just remove profile_id)
    cur.execute("UPDATE speeches SET profile_id = NULL WHERE profile_id IN (SELECT id FROM care_profiles WHERE group_id = %s)", (group_id,))
    
    # Delete group (cascades to members, invites, profiles)
    cur.execute("DELETE FROM care_groups WHERE id = %s", (group_id,))
    
    conn.commit()
    conn.close()
    
    logger.info(f"User {user_id} deleted group {group_id}")
    return {"message": "Group deleted"}


# --- Member Endpoints ---

@router.get("/{group_id}/members")
async def list_members(group_id: str, auth: dict = Depends(flexible_auth)):
    """List all members in a group."""
    user_id = auth.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    require_group_role(user_id, group_id, ['admin', 'caregiver', 'viewer'])
    
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('''
        SELECT m.*, u.email, u.name as user_name
        FROM group_members m
        JOIN users u ON u.id = m.user_id
        WHERE m.group_id = %s
        ORDER BY m.joined_at
    ''', (group_id,))
    members = cur.fetchall()
    conn.close()
    
    return {"members": [dict(m) for m in members]}


@router.post("/{group_id}/invite")
async def create_invite(group_id: str, data: InviteCreate, auth: dict = Depends(flexible_auth)):
    """Create an invite code for someone to join the group."""
    user_id = auth.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    require_group_role(user_id, group_id, ['admin'])
    
    if data.role not in ('admin', 'caregiver', 'viewer'):
        raise HTTPException(status_code=400, detail="Role must be: admin, caregiver, or viewer")
    
    invite_code = generate_invite_code()
    
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO group_invites (group_id, invite_code, email, role, invited_by)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id, invite_code, role, email, expires_at
    ''', (group_id, invite_code, data.email, data.role, user_id))
    invite = dict(cur.fetchone())
    conn.commit()
    conn.close()
    
    logger.info(f"User {user_id} created invite {invite_code} for group {group_id}")
    return {"invite": invite}


@router.post("/join")
async def join_group(data: JoinGroup, auth: dict = Depends(flexible_auth)):
    """Join a group using an invite code."""
    user_id = auth.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Find the invite
    cur.execute('''
        SELECT * FROM group_invites 
        WHERE invite_code = %s AND status = 'pending' AND expires_at > NOW()
    ''', (data.invite_code.upper(),))
    invite = cur.fetchone()
    
    if not invite:
        conn.close()
        raise HTTPException(status_code=404, detail="Invalid or expired invite code")
    
    # Check email restriction
    if invite['email']:
        cur.execute("SELECT email FROM users WHERE id = %s", (user_id,))
        user = cur.fetchone()
        if user['email'].lower() != invite['email'].lower():
            conn.close()
            raise HTTPException(status_code=403, detail="This invite is for a different email address")
    
    # Check if already a member
    cur.execute(
        "SELECT 1 FROM group_members WHERE group_id = %s AND user_id = %s",
        (invite['group_id'], user_id)
    )
    if cur.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="You're already a member of this group")
    
    # Add as member
    cur.execute('''
        INSERT INTO group_members (group_id, user_id, role, invited_by)
        VALUES (%s, %s, %s, %s)
    ''', (invite['group_id'], user_id, invite['role'], invite['invited_by']))
    
    # Mark invite as accepted
    cur.execute('''
        UPDATE group_invites SET status = 'accepted', accepted_at = NOW(), accepted_by = %s
        WHERE id = %s
    ''', (user_id, invite['id']))
    
    # Get group info
    cur.execute("SELECT name FROM care_groups WHERE id = %s", (invite['group_id'],))
    group = cur.fetchone()
    
    conn.commit()
    conn.close()
    
    logger.info(f"User {user_id} joined group {invite['group_id']} as {invite['role']}")
    return {"message": f"Joined '{group['name']}' as {invite['role']}", "group_id": str(invite['group_id'])}


@router.delete("/{group_id}/members/{member_user_id}")
async def remove_member(group_id: str, member_user_id: int, auth: dict = Depends(flexible_auth)):
    """Remove a member from the group (admin only)."""
    user_id = auth.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    require_group_role(user_id, group_id, ['admin'])
    
    # Can't remove yourself if you're the owner
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT owner_id FROM care_groups WHERE id = %s", (group_id,))
    group = cur.fetchone()
    
    if group['owner_id'] == member_user_id:
        conn.close()
        raise HTTPException(status_code=400, detail="Cannot remove the group owner")
    
    cur.execute(
        "DELETE FROM group_members WHERE group_id = %s AND user_id = %s",
        (group_id, member_user_id)
    )
    conn.commit()
    conn.close()
    
    return {"message": "Member removed"}


# --- Profile Endpoints ---

@router.get("/{group_id}/profiles")
async def list_profiles(group_id: str, auth: dict = Depends(flexible_auth)):
    """List all care profiles in a group."""
    user_id = auth.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    require_group_role(user_id, group_id, ['admin', 'caregiver', 'viewer'])
    
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('''
        SELECT p.*,
               (SELECT COUNT(*) FROM speeches WHERE profile_id = p.id) as recording_count,
               (SELECT MAX(created_at) FROM speeches WHERE profile_id = p.id) as last_recording
        FROM care_profiles p
        WHERE p.group_id = %s
        ORDER BY p.name
    ''', (group_id,))
    profiles = cur.fetchall()
    conn.close()
    
    return {"profiles": [dict(p) for p in profiles]}


@router.post("/{group_id}/profiles")
async def create_profile(group_id: str, data: ProfileCreate, auth: dict = Depends(flexible_auth)):
    """Create a new care profile (admin or caregiver)."""
    user_id = auth.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    require_group_role(user_id, group_id, ['admin', 'caregiver'])
    
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO care_profiles (group_id, name, avatar_emoji, avatar_color, 
                                   date_of_birth, diagnosis, notes, 
                                   physician_name, physician_email, created_by)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING *
    ''', (group_id, data.name, data.avatar_emoji, data.avatar_color,
          data.date_of_birth, data.diagnosis, data.notes,
          data.physician_name, data.physician_email, user_id))
    profile = dict(cur.fetchone())
    conn.commit()
    conn.close()
    
    logger.info(f"User {user_id} created profile '{data.name}' in group {group_id}")
    return {"profile": profile}


@router.get("/profiles/{profile_id}")
async def get_profile(profile_id: str, auth: dict = Depends(flexible_auth)):
    """Get a care profile's details."""
    user_id = auth.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Get profile and verify access
    cur.execute('''
        SELECT p.*, g.name as group_name
        FROM care_profiles p
        JOIN care_groups g ON g.id = p.group_id
        WHERE p.id = %s
    ''', (profile_id,))
    profile = cur.fetchone()
    
    if not profile:
        conn.close()
        raise HTTPException(status_code=404, detail="Profile not found")
    
    # Check membership
    cur.execute(
        "SELECT 1 FROM group_members WHERE group_id = %s AND user_id = %s",
        (profile['group_id'], user_id)
    )
    if not cur.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Profile not found")
    
    conn.close()
    return {"profile": dict(profile)}


@router.patch("/profiles/{profile_id}")
async def update_profile(profile_id: str, data: ProfileUpdate, auth: dict = Depends(flexible_auth)):
    """Update a care profile."""
    user_id = auth.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Get profile's group
    cur.execute("SELECT group_id FROM care_profiles WHERE id = %s", (profile_id,))
    profile = cur.fetchone()
    if not profile:
        conn.close()
        raise HTTPException(status_code=404, detail="Profile not found")
    
    require_group_role(user_id, str(profile['group_id']), ['admin', 'caregiver'])
    
    updates = []
    values = []
    for field in ['name', 'avatar_emoji', 'avatar_color', 'date_of_birth', 
                  'diagnosis', 'notes', 'physician_name', 'physician_email', 'settings']:
        value = getattr(data, field, None)
        if value is not None:
            updates.append(f"{field} = %s")
            values.append(value)
    
    if not updates:
        conn.close()
        raise HTTPException(status_code=400, detail="No fields to update")
    
    updates.append("updated_at = NOW()")
    values.append(profile_id)
    
    cur.execute(f'''
        UPDATE care_profiles SET {", ".join(updates)}
        WHERE id = %s
        RETURNING *
    ''', values)
    updated = dict(cur.fetchone())
    conn.commit()
    conn.close()
    
    return {"profile": updated}


@router.delete("/profiles/{profile_id}")
async def delete_profile(profile_id: str, auth: dict = Depends(flexible_auth)):
    """Delete a care profile (admin only). Recordings will be unlinked."""
    user_id = auth.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Get profile's group
    cur.execute("SELECT group_id FROM care_profiles WHERE id = %s", (profile_id,))
    profile = cur.fetchone()
    if not profile:
        conn.close()
        raise HTTPException(status_code=404, detail="Profile not found")
    
    require_group_role(user_id, str(profile['group_id']), ['admin'])
    
    # Unlink recordings
    cur.execute("UPDATE speeches SET profile_id = NULL WHERE profile_id = %s", (profile_id,))
    
    # Delete profile
    cur.execute("DELETE FROM care_profiles WHERE id = %s", (profile_id,))
    
    conn.commit()
    conn.close()
    
    logger.info(f"User {user_id} deleted profile {profile_id}")
    return {"message": "Profile deleted"}


@router.get("/profiles/{profile_id}/recordings")
async def get_profile_recordings(
    profile_id: str,
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0),
    auth: dict = Depends(flexible_auth)
):
    """Get recordings for a specific care profile."""
    user_id = auth.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Get profile and verify access
    cur.execute("SELECT group_id FROM care_profiles WHERE id = %s", (profile_id,))
    profile = cur.fetchone()
    if not profile:
        conn.close()
        raise HTTPException(status_code=404, detail="Profile not found")
    
    cur.execute(
        "SELECT 1 FROM group_members WHERE group_id = %s AND user_id = %s",
        (profile['group_id'], user_id)
    )
    if not cur.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Profile not found")
    
    # Get recordings
    cur.execute('''
        SELECT id, title, duration_sec, created_at, dementia_metrics
        FROM speeches
        WHERE profile_id = %s
        ORDER BY created_at DESC
        LIMIT %s OFFSET %s
    ''', (profile_id, limit, offset))
    recordings = cur.fetchall()
    
    # Get total count
    cur.execute("SELECT COUNT(*) as total FROM speeches WHERE profile_id = %s", (profile_id,))
    total = cur.fetchone()['total']
    
    conn.close()
    
    return {
        "recordings": [dict(r) for r in recordings],
        "total": total,
        "limit": limit,
        "offset": offset
    }


@router.get("/profiles/{profile_id}/metrics")
async def get_profile_metrics(
    profile_id: str,
    days: int = Query(default=30, le=365),
    auth: dict = Depends(flexible_auth)
):
    """Get aggregated metrics for a care profile over time."""
    user_id = auth.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Verify access
    cur.execute("SELECT group_id FROM care_profiles WHERE id = %s", (profile_id,))
    profile = cur.fetchone()
    if not profile:
        conn.close()
        raise HTTPException(status_code=404, detail="Profile not found")
    
    cur.execute(
        "SELECT 1 FROM group_members WHERE group_id = %s AND user_id = %s",
        (profile['group_id'], user_id)
    )
    if not cur.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Profile not found")
    
    # Get metrics over time
    cur.execute('''
        SELECT 
            DATE(created_at) as date,
            COUNT(*) as recording_count,
            AVG((dementia_metrics->>'composite_score')::float) as avg_score,
            AVG((dementia_metrics->>'word_finding_difficulty')::float) as avg_word_finding,
            AVG((dementia_metrics->>'repetition_rate')::float) as avg_repetition
        FROM speeches
        WHERE profile_id = %s 
          AND created_at > NOW() - INTERVAL '%s days'
          AND dementia_metrics IS NOT NULL
        GROUP BY DATE(created_at)
        ORDER BY date DESC
    ''', (profile_id, days))
    daily_metrics = cur.fetchall()
    
    conn.close()
    
    return {
        "profile_id": profile_id,
        "days": days,
        "metrics": [dict(m) for m in daily_metrics]
    }
