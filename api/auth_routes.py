"""
Authentication route handlers — social login, token refresh, user profile.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Depends, Query
from pydantic import BaseModel

from . import config
from .user_auth import (
    verify_google_token,
    get_or_create_user,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_user_by_id,
    user_required,
    update_user_app_version,
    get_all_users_with_versions,
)

logger = logging.getLogger("speechscore.auth_routes")

auth_router = APIRouter(prefix="/v1/auth", tags=["Authentication"])

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class GoogleLoginRequest(BaseModel):
    id_token: str

class AppleLoginRequest(BaseModel):
    identity_token: str
    authorization_code: Optional[str] = None
    name: Optional[str] = None  # Apple only sends name on first login

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int
    user: dict

class RefreshRequest(BaseModel):
    refresh_token: str

class UserProfile(BaseModel):
    id: int
    email: Optional[str]
    name: Optional[str]
    avatar_url: Optional[str]
    provider: str
    created_at: str


# ---------------------------------------------------------------------------
# Google Sign-In
# ---------------------------------------------------------------------------

@auth_router.post("/google", response_model=TokenResponse)
async def google_login(body: GoogleLoginRequest):
    """
    Authenticate with Google Sign-In.
    
    Send the Google ID token from the mobile SDK.
    Returns JWT access + refresh tokens.
    """
    # Verify the Google ID token
    google_user = verify_google_token(body.id_token)
    if not google_user:
        raise HTTPException(
            status_code=401,
            detail={"code": "INVALID_TOKEN", "message": "Invalid or expired Google ID token."}
        )

    # Get or create user in our database
    db_path = config.DB_PATH
    user = get_or_create_user(
        db_path=db_path,
        provider="google",
        provider_id=google_user["sub"],
        email=google_user.get("email"),
        name=google_user.get("name"),
        avatar_url=google_user.get("picture"),
    )

    # Check if account is disabled
    if user.get("enabled") == 0:
        raise HTTPException(
            status_code=403,
            detail={"code": "ACCOUNT_DISABLED", "message": "Your account has been disabled. Contact support for assistance."}
        )

    # Issue JWT tokens
    access_token = create_access_token(user["id"], user.get("email", ""), user.get("name", ""))
    refresh_token = create_refresh_token(user["id"])

    logger.info("Google login: user_id=%d, email=%s", user["id"], user.get("email"))

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=7 * 24 * 3600,  # 7 days
        user={
            "id": user["id"],
            "email": user.get("email"),
            "name": user.get("name"),
            "avatar_url": user.get("avatar_url"),
            "provider": "google",
            "tier": user.get("tier", "free"),
        },
    )


# ---------------------------------------------------------------------------
# Apple Sign-In (placeholder — needs Apple Developer account)
# ---------------------------------------------------------------------------

@auth_router.post("/apple", response_model=TokenResponse)
async def apple_login(body: AppleLoginRequest):
    """
    Authenticate with Apple Sign-In.
    
    Send the Apple identity token from the mobile SDK.
    Returns JWT access + refresh tokens.
    
    NOTE: Requires Apple Developer account configuration in oauth_config.json.
    """
    raise HTTPException(
        status_code=501,
        detail={"code": "NOT_IMPLEMENTED", "message": "Apple Sign-In not yet configured. Set up Apple Developer credentials in oauth_config.json."}
    )


# ---------------------------------------------------------------------------
# Token Refresh
# ---------------------------------------------------------------------------

@auth_router.post("/refresh", response_model=TokenResponse)
async def refresh_token(body: RefreshRequest):
    """
    Refresh an expired access token using a valid refresh token.
    """
    payload = decode_token(body.refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=401,
            detail={"code": "INVALID_TOKEN", "message": "Invalid or expired refresh token."}
        )

    user_id = int(payload["sub"])
    db_path = config.DB_PATH
    user = get_user_by_id(db_path, user_id)

    if not user:
        raise HTTPException(
            status_code=401,
            detail={"code": "USER_NOT_FOUND", "message": "User no longer exists."}
        )

    # Check if account is disabled
    if user.get("enabled") == 0:
        raise HTTPException(
            status_code=403,
            detail={"code": "ACCOUNT_DISABLED", "message": "Your account has been disabled. Contact support for assistance."}
        )

    access_token = create_access_token(user["id"], user.get("email", ""), user.get("name", ""))
    new_refresh = create_refresh_token(user["id"])

    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh,
        expires_in=7 * 24 * 3600,
        user={
            "id": user["id"],
            "email": user.get("email"),
            "name": user.get("name"),
            "avatar_url": user.get("avatar_url"),
            "provider": user["provider"],
        },
    )


# ---------------------------------------------------------------------------
# User Profile
# ---------------------------------------------------------------------------

@auth_router.get("/me")
async def get_profile(user: dict = Depends(user_required)):
    """Get the current user's profile."""
    db_path = config.DB_PATH
    full_user = get_user_by_id(db_path, user["id"])

    if not full_user:
        raise HTTPException(
            status_code=404,
            detail={"code": "USER_NOT_FOUND", "message": "User not found."}
        )

    return {
        "id": full_user["id"],
        "email": full_user.get("email"),
        "name": full_user.get("name"),
        "avatar_url": full_user.get("avatar_url"),
        "provider": full_user["provider"],
        "created_at": full_user["created_at"],
        "last_login": full_user["last_login"],
        "tier": full_user.get("tier", "free"),
    }


@auth_router.delete("/me")
async def delete_account(user: dict = Depends(user_required)):
    """Delete the current user's account and all their data."""
    import sqlite3
    db_path = config.DB_PATH
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    user_id = user["id"]

    # Delete user's speeches and associated data
    cur.execute("SELECT id FROM speeches WHERE user_id = ?", (user_id,))
    speech_ids = [row[0] for row in cur.fetchall()]

    for sid in speech_ids:
        cur.execute("DELETE FROM analyses WHERE speech_id = ?", (sid,))
        cur.execute("DELETE FROM transcriptions WHERE speech_id = ?", (sid,))
        cur.execute("DELETE FROM audio WHERE speech_id = ?", (sid,))
        # Delete spectrograms if table exists
        try:
            cur.execute("DELETE FROM spectrograms WHERE speech_id = ?", (sid,))
        except sqlite3.OperationalError:
            pass

    cur.execute("DELETE FROM speeches WHERE user_id = ?", (user_id,))
    cur.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()

    logger.info("Deleted user account: user_id=%d, speeches=%d", user_id, len(speech_ids))

    return {"message": "Account and all associated data deleted.", "speeches_deleted": len(speech_ids)}


# ---------------------------------------------------------------------------
# App Version Tracking
# ---------------------------------------------------------------------------

class AppVersionReport(BaseModel):
    app_version: str
    patch_number: Optional[int] = None


@auth_router.post("/version")
async def report_app_version(body: AppVersionReport, user: dict = Depends(user_required)):
    """
    Report the current app version and Shorebird patch number.
    Called on app startup after login.
    """
    db_path = config.DB_PATH
    update_user_app_version(
        db_path=db_path,
        user_id=user["id"],
        app_version=body.app_version,
        patch_number=body.patch_number,
    )
    logger.info("Version report: user_id=%d, version=%s, patch=%s", 
                user["id"], body.app_version, body.patch_number)
    return {"status": "ok"}


@auth_router.get("/admin/users/versions")
async def get_user_versions(api_key: str = Query(None)):
    """
    Admin endpoint: Get all users with their app versions.
    Requires API key for access (not user JWT).
    """
    from .auth import lookup_key
    
    # Verify API key
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")
    
    key_data = lookup_key(api_key)
    if not key_data or not key_data.get("enabled"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    db_path = config.DB_PATH
    users = get_all_users_with_versions(db_path)
    
    return {
        "users": users,
        "total": len(users),
    }


# ---------------------------------------------------------------------------
# Usage / Rate Limits
# ---------------------------------------------------------------------------

@auth_router.get("/usage")
async def get_usage(user: dict = Depends(user_required)):
    """
    Get current usage statistics and limits for the logged-in user.
    
    Returns:
    - tier: Current subscription tier (free, pro, power, admin)
    - recordings_used: Recordings this month
    - recordings_limit: Monthly limit (or "unlimited")
    - recordings_remaining: Remaining recordings
    - audio_minutes_used: Audio minutes processed this month
    - audio_minutes_limit: Monthly limit (or "unlimited")
    - audio_minutes_remaining: Remaining minutes
    - resets: Date when usage resets (first of next month)
    - features: List of features available on this tier
    """
    import sqlite3
    from .usage_limits import get_usage_limiter
    
    db_path = config.DB_PATH
    conn = sqlite3.connect(db_path)
    
    usage_limiter = get_usage_limiter(conn)
    usage = usage_limiter.get_user_usage(user["id"])
    
    conn.close()
    
    return usage


@auth_router.post("/usage/accept-eula")
async def accept_eula(user: dict = Depends(user_required)):
    """Record that the user accepted the EULA."""
    import sqlite3
    from .usage_limits import get_usage_limiter
    
    db_path = config.DB_PATH
    conn = sqlite3.connect(db_path)
    
    usage_limiter = get_usage_limiter(conn)
    usage_limiter.record_eula_acceptance(user["id"])
    
    conn.close()
    
    logger.info("EULA accepted: user_id=%d", user["id"])
    return {"status": "ok", "message": "EULA acceptance recorded"}


@auth_router.post("/admin/users/{user_id}/tier")
async def set_user_tier(
    user_id: int, 
    tier: str = Query(...),
    user: dict = Depends(user_required),
):
    """
    Admin endpoint: Set a user's subscription tier.
    Requires admin tier.
    
    Valid tiers: free, pro, power, admin
    """
    from .usage_limits import get_usage_limiter, TIER_LIMITS
    import sqlite3
    
    # Verify admin access
    _require_admin(user)
    
    # Validate tier
    if tier not in TIER_LIMITS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid tier. Valid tiers: {', '.join(TIER_LIMITS.keys())}"
        )
    
    db_path = config.DB_PATH
    conn = sqlite3.connect(db_path)
    
    usage_limiter = get_usage_limiter(conn)
    success = usage_limiter.set_user_tier(user_id, tier)
    
    conn.close()
    
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    
    logger.info("Set tier: user_id=%d, tier=%s", user_id, tier)
    return {"status": "ok", "user_id": user_id, "tier": tier}


# ============================================================================
# Mobile Admin API Endpoints
# ============================================================================

def _require_admin(user: dict):
    """Check if user has admin tier (fetches from DB)."""
    import sqlite3
    db_path = config.DB_PATH
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT tier FROM users WHERE id = ?", (user["id"],))
    row = cur.fetchone()
    conn.close()
    
    if not row or row[0] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")


@auth_router.get("/admin/users")
async def admin_list_users(
    user: dict = Depends(user_required),
    search: str = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
):
    """
    List all users with search and pagination.
    Requires admin tier.
    """
    _require_admin(user)
    
    import sqlite3
    db_path = config.DB_PATH
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    offset = (page - 1) * limit
    
    # Build query
    if search:
        search_pattern = f"%{search}%"
        cur.execute("""
            SELECT id, email, name, avatar_url, provider, tier, 
                   monthly_recordings, monthly_audio_minutes, created_at, last_login,
                   (SELECT COUNT(*) FROM speeches WHERE user_id = users.id) as speech_count
            FROM users 
            WHERE email LIKE ? OR name LIKE ?
            ORDER BY last_login DESC
            LIMIT ? OFFSET ?
        """, (search_pattern, search_pattern, limit, offset))
        
        cur2 = conn.cursor()
        cur2.execute("""
            SELECT COUNT(*) FROM users WHERE email LIKE ? OR name LIKE ?
        """, (search_pattern, search_pattern))
        total = cur2.fetchone()[0]
    else:
        cur.execute("""
            SELECT id, email, name, avatar_url, provider, tier,
                   monthly_recordings, monthly_audio_minutes, created_at, last_login,
                   (SELECT COUNT(*) FROM speeches WHERE user_id = users.id) as speech_count
            FROM users 
            ORDER BY last_login DESC
            LIMIT ? OFFSET ?
        """, (limit, offset))
        
        cur2 = conn.cursor()
        cur2.execute("SELECT COUNT(*) FROM users")
        total = cur2.fetchone()[0]
    
    users = [dict(row) for row in cur.fetchall()]
    conn.close()
    
    return {
        "users": users,
        "total": total,
        "page": page,
        "limit": limit,
        "pages": (total + limit - 1) // limit,
    }


@auth_router.get("/admin/users/{user_id}")
async def admin_get_user(
    user_id: int,
    user: dict = Depends(user_required),
):
    """
    Get detailed user info.
    Requires admin tier.
    """
    _require_admin(user)
    
    import sqlite3
    db_path = config.DB_PATH
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    cur.execute("""
        SELECT id, email, name, avatar_url, provider, tier,
               monthly_recordings, monthly_audio_minutes, usage_month,
               created_at, last_login, eula_accepted_at, disabled
        FROM users WHERE id = ?
    """, (user_id,))
    row = cur.fetchone()
    
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")
    
    target_user = dict(row)
    
    # Convert disabled integer to boolean
    target_user["disabled"] = bool(target_user.get("disabled", 0))
    
    # Get speech stats
    cur.execute("""
        SELECT COUNT(*) as count, 
               COALESCE(SUM(duration_sec), 0) as total_duration
        FROM speeches WHERE user_id = ?
    """, (user_id,))
    stats = dict(cur.fetchone())
    target_user["speech_count"] = stats["count"]
    target_user["total_duration_seconds"] = stats["total_duration"]
    
    # Get recent speeches
    cur.execute("""
        SELECT id, title, profile, duration_sec, created_at
        FROM speeches WHERE user_id = ?
        ORDER BY created_at DESC LIMIT 10
    """, (user_id,))
    target_user["recent_speeches"] = [dict(r) for r in cur.fetchall()]
    
    conn.close()
    return target_user


@auth_router.post("/admin/users/{user_id}/disable")
async def admin_disable_user(
    user_id: int,
    disabled: bool = Query(...),
    user: dict = Depends(user_required),
):
    """
    Enable or disable a user account.
    Disabled users cannot log in.
    Requires admin tier.
    """
    _require_admin(user)
    
    import sqlite3
    db_path = config.DB_PATH
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    cur.execute("UPDATE users SET disabled = ? WHERE id = ?", (1 if disabled else 0, user_id))
    if cur.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")
    
    conn.commit()
    conn.close()
    
    logger.info("User %d %s by admin %d", user_id, "disabled" if disabled else "enabled", user["id"])
    return {"status": "ok", "user_id": user_id, "disabled": disabled}


@auth_router.post("/admin/users/{user_id}/reset-usage")
async def admin_reset_user_usage(
    user_id: int,
    user: dict = Depends(user_required),
):
    """
    Reset a user's monthly usage counters to zero.
    Requires admin tier.
    """
    _require_admin(user)
    
    import sqlite3
    db_path = config.DB_PATH
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    cur.execute("""
        UPDATE users 
        SET monthly_recordings = 0, monthly_audio_minutes = 0
        WHERE id = ?
    """, (user_id,))
    
    if cur.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")
    
    conn.commit()
    conn.close()
    
    logger.info("Admin reset usage: user_id=%d by admin=%d", user_id, user["id"])
    return {"status": "ok", "message": "Usage reset to zero"}


@auth_router.post("/admin/users/{user_id}/recalc-usage")
async def admin_recalc_user_usage(
    user_id: int,
    user: dict = Depends(user_required),
):
    """
    Recalculate a user's monthly usage from their actual recordings.
    Requires admin tier.
    """
    _require_admin(user)
    
    import sqlite3
    from datetime import datetime
    
    db_path = config.DB_PATH
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    current_month = datetime.now().strftime("%Y-%m")
    
    # Count recordings this month
    cur.execute("""
        SELECT COUNT(*), COALESCE(SUM(duration_sec) / 60.0, 0)
        FROM speeches 
        WHERE user_id = ? AND strftime('%Y-%m', created_at) = ?
    """, (user_id, current_month))
    
    row = cur.fetchone()
    recordings = row[0]
    audio_minutes = row[1]
    
    cur.execute("""
        UPDATE users 
        SET monthly_recordings = ?, monthly_audio_minutes = ?, usage_month = ?
        WHERE id = ?
    """, (recordings, audio_minutes, current_month, user_id))
    
    if cur.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")
    
    conn.commit()
    conn.close()
    
    logger.info("Admin recalc usage: user_id=%d recordings=%d minutes=%.1f", 
                user_id, recordings, audio_minutes)
    return {
        "status": "ok", 
        "monthly_recordings": recordings,
        "monthly_audio_minutes": round(audio_minutes, 1),
    }


@auth_router.delete("/admin/users/{user_id}")
async def admin_delete_user(
    user_id: int,
    user: dict = Depends(user_required),
):
    """
    Delete a user and all their data.
    Requires admin tier. Cannot delete yourself.
    """
    _require_admin(user)
    
    if user_id == user["id"]:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    
    import sqlite3
    db_path = config.DB_PATH
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Check user exists
    cur.execute("SELECT email FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")
    
    target_email = row[0]
    
    # Delete speeches and analyses
    cur.execute("SELECT id FROM speeches WHERE user_id = ?", (user_id,))
    speech_ids = [r[0] for r in cur.fetchall()]
    
    for sid in speech_ids:
        cur.execute("DELETE FROM analyses WHERE speech_id = ?", (sid,))
    
    cur.execute("DELETE FROM speeches WHERE user_id = ?", (user_id,))
    cur.execute("DELETE FROM speaker_embeddings WHERE user_id = ?", (user_id,))
    cur.execute("DELETE FROM projects WHERE user_id = ?", (user_id,))
    cur.execute("DELETE FROM users WHERE id = ?", (user_id,))
    
    conn.commit()
    conn.close()
    
    logger.info("Admin deleted user: id=%d email=%s by admin=%d", 
                user_id, target_email, user["id"])
    return {"status": "ok", "message": f"Deleted user {target_email}"}
