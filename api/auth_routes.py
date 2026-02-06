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
