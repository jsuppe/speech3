"""
Simple session-based authentication for the frontend dashboard.
Only allows specific email addresses to access the dashboard.
"""

import os
import secrets
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Request, HTTPException
from fastapi.responses import RedirectResponse

logger = logging.getLogger("speechscore.frontend.auth")

# Allowed email addresses (only these can log in)
ALLOWED_EMAILS = {
    "jon.suppe@gmail.com",
}

# Session storage (in-memory for simplicity)
# In production, use Redis or database
_sessions = {}

# Session cookie name
SESSION_COOKIE = "speechscore_dashboard_session"
SESSION_DURATION_HOURS = 24 * 7  # 1 week


def generate_session_token() -> str:
    """Generate a secure random session token."""
    return secrets.token_urlsafe(32)


def create_session(email: str) -> str:
    """Create a new session for the given email."""
    token = generate_session_token()
    _sessions[token] = {
        "email": email,
        "created_at": datetime.utcnow(),
        "expires_at": datetime.utcnow() + timedelta(hours=SESSION_DURATION_HOURS),
    }
    logger.info(f"Created session for {email}")
    return token


def get_session(token: str) -> Optional[dict]:
    """Get session data if token is valid and not expired."""
    if not token or token not in _sessions:
        return None
    
    session = _sessions[token]
    if datetime.utcnow() > session["expires_at"]:
        # Session expired, remove it
        del _sessions[token]
        return None
    
    return session


def destroy_session(token: str):
    """Remove a session."""
    if token in _sessions:
        del _sessions[token]


def is_email_allowed(email: str) -> bool:
    """Check if the email is in the allowed list."""
    return email.lower() in {e.lower() for e in ALLOWED_EMAILS}


def get_current_user(request: Request) -> Optional[dict]:
    """Get current user from session cookie."""
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    return get_session(token)


def require_auth(request: Request) -> dict:
    """Require authentication. Returns user dict or raises redirect."""
    user = get_current_user(request)
    if not user:
        # Redirect to login
        raise HTTPException(status_code=307, headers={"Location": "/login"})
    return user


async def verify_google_token(id_token: str) -> Optional[dict]:
    """Verify Google ID token and return user info."""
    import httpx
    
    try:
        # Verify with Google's tokeninfo endpoint
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://oauth2.googleapis.com/tokeninfo?id_token={id_token}"
            )
            
            if response.status_code != 200:
                logger.warning(f"Google token verification failed: {response.text}")
                return None
            
            data = response.json()
            
            # Check if token is valid
            if "error" in data:
                logger.warning(f"Google token error: {data['error']}")
                return None
            
            return {
                "email": data.get("email"),
                "name": data.get("name"),
                "picture": data.get("picture"),
            }
    except Exception as e:
        logger.exception(f"Error verifying Google token: {e}")
        return None
