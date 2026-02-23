"""
User authentication system for SpeechScore API.

Supports:
- Google Sign-In (ID token verification)
- Apple Sign-In (identity token verification) — placeholder
- JWT token issuance and verification

Works alongside the existing API key system.
"""

import os
import jwt
import json
import time
import sqlite3
import logging
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, Tuple

from fastapi import Request, Response, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests

logger = logging.getLogger("speechscore.user_auth")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# JWT settings
JWT_SECRET = os.environ.get("SPEECHSCORE_JWT_SECRET", None)
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_HOURS = 24 * 7  # 7 days
JWT_REFRESH_TOKEN_EXPIRE_DAYS = 90

# Google OAuth
GOOGLE_CLIENT_IDS = []  # Populated from config
_google_client_ids_loaded = False

# Config file for OAuth settings
_AUTH_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oauth_config.json")

# Default config template
_DEFAULT_AUTH_CONFIG = {
    "google": {
        "client_ids": [],
        "enabled": False,
    },
    "apple": {
        "team_id": "",
        "key_id": "",
        "bundle_id": "",
        "enabled": False,
    },
    "jwt_secret": "",  # Auto-generated on first run if empty
}


def _load_auth_config() -> dict:
    """Load OAuth configuration."""
    if os.path.exists(_AUTH_CONFIG_FILE):
        with open(_AUTH_CONFIG_FILE) as f:
            return json.load(f)
    return _DEFAULT_AUTH_CONFIG.copy()


def _save_auth_config(config: dict):
    """Save OAuth configuration."""
    with open(_AUTH_CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def init_user_auth(db_path: str):
    """
    Initialize user auth system:
    - Generate JWT secret if needed
    - Load OAuth config
    - Create users table if needed
    """
    global JWT_SECRET, GOOGLE_CLIENT_IDS, _google_client_ids_loaded

    config = _load_auth_config()

    # Generate JWT secret on first run
    if not config.get("jwt_secret"):
        config["jwt_secret"] = secrets.token_hex(32)
        _save_auth_config(config)
        logger.info("Generated new JWT secret.")

    # Use env var if set, otherwise config file
    if not JWT_SECRET:
        JWT_SECRET = config["jwt_secret"]

    # Load Google client IDs
    google_cfg = config.get("google", {})
    GOOGLE_CLIENT_IDS = google_cfg.get("client_ids", [])
    _google_client_ids_loaded = True

    if GOOGLE_CLIENT_IDS:
        logger.info("Google Sign-In enabled with %d client ID(s).", len(GOOGLE_CLIENT_IDS))
    else:
        logger.info("Google Sign-In not configured (no client_ids in oauth_config.json).")

    # Create/migrate users table
    _ensure_users_table(db_path)

    logger.info("User auth system initialized.")


def _ensure_users_table(db_path: str):
    """Create users table and add user_id to speeches if needed."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Create users table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            provider TEXT NOT NULL,          -- 'google', 'apple', 'email'
            provider_id TEXT NOT NULL,       -- Provider's unique user ID
            email TEXT,
            name TEXT,
            avatar_url TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            last_login TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            app_version TEXT,                -- e.g., '1.0.22+23'
            patch_number INTEGER,            -- Shorebird patch number
            last_seen TEXT,                  -- Last app heartbeat timestamp
            UNIQUE(provider, provider_id)
        )
    """)
    
    # Add version tracking columns if they don't exist
    cur.execute("PRAGMA table_info(users)")
    columns = [row[1] for row in cur.fetchall()]
    if "app_version" not in columns:
        cur.execute("ALTER TABLE users ADD COLUMN app_version TEXT")
        logger.info("Added app_version column to users table.")
    if "patch_number" not in columns:
        cur.execute("ALTER TABLE users ADD COLUMN patch_number INTEGER")
        logger.info("Added patch_number column to users table.")
    if "last_seen" not in columns:
        cur.execute("ALTER TABLE users ADD COLUMN last_seen TEXT")
        logger.info("Added last_seen column to users table.")

    # Add user_id column to speeches if it doesn't exist
    cur.execute("PRAGMA table_info(speeches)")
    columns = [row[1] for row in cur.fetchall()]
    if "user_id" not in columns:
        cur.execute("ALTER TABLE speeches ADD COLUMN user_id INTEGER REFERENCES users(id)")
        logger.info("Added user_id column to speeches table.")

    # Update schema version
    cur.execute("SELECT MAX(version) FROM schema_version")
    current = cur.fetchone()[0] or 0
    if current < 2:
        cur.execute("INSERT INTO schema_version (version, applied_at) VALUES (2, ?)",
                    (datetime.now(timezone.utc).isoformat(),))

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# JWT Token Management
# ---------------------------------------------------------------------------

def create_access_token(user_id: int, email: str, name: str) -> str:
    """Create a JWT access token."""
    payload = {
        "sub": str(user_id),
        "email": email,
        "name": name,
        "type": "access",
        "iat": int(time.time()),
        "exp": int(time.time()) + JWT_ACCESS_TOKEN_EXPIRE_HOURS * 3600,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def create_refresh_token(user_id: int) -> str:
    """Create a JWT refresh token."""
    payload = {
        "sub": str(user_id),
        "type": "refresh",
        "iat": int(time.time()),
        "exp": int(time.time()) + JWT_REFRESH_TOKEN_EXPIRE_DAYS * 86400,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and verify a JWT token. Returns payload or None."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


# ---------------------------------------------------------------------------
# Google Sign-In Verification
# ---------------------------------------------------------------------------

def verify_google_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify a Google ID token.
    Returns user info dict: {sub, email, name, picture} or None.
    """
    try:
        # If client IDs are configured, verify audience
        if GOOGLE_CLIENT_IDS:
            idinfo = google_id_token.verify_oauth2_token(
                token,
                google_requests.Request(),
                clock_skew_in_seconds=10,
            )
            # Check audience matches one of our client IDs
            if idinfo.get("aud") not in GOOGLE_CLIENT_IDS:
                logger.warning("Google token audience mismatch: %s", idinfo.get("aud"))
                return None
        else:
            # No client IDs configured — verify structure but accept any audience
            # This is for development/testing
            idinfo = google_id_token.verify_oauth2_token(
                token,
                google_requests.Request(),
                clock_skew_in_seconds=10,
            )

        if not idinfo.get("email_verified", False):
            logger.warning("Google email not verified for: %s", idinfo.get("email"))
            return None

        return {
            "sub": idinfo["sub"],
            "email": idinfo.get("email"),
            "name": idinfo.get("name", ""),
            "picture": idinfo.get("picture", ""),
        }
    except ValueError as e:
        logger.warning("Google token verification failed: %s", e)
        return None
    except Exception as e:
        logger.error("Unexpected error verifying Google token: %s", e)
        return None


# ---------------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------------

def get_or_create_user(db_path: str, provider: str, provider_id: str,
                       email: str = None, name: str = None,
                       avatar_url: str = None) -> Dict[str, Any]:
    """
    Find existing user by provider+provider_id, or create a new one.
    Returns user dict with id, provider, email, name, etc.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    now = datetime.now(timezone.utc).isoformat()

    # Try to find existing user
    cur.execute(
        "SELECT * FROM users WHERE provider = ? AND provider_id = ?",
        (provider, provider_id)
    )
    row = cur.fetchone()

    if row:
        user = dict(row)
        # Update last_login and any changed fields
        updates = {"last_login": now}
        if email and email != user.get("email"):
            updates["email"] = email
        if name and name != user.get("name"):
            updates["name"] = name
        if avatar_url and avatar_url != user.get("avatar_url"):
            updates["avatar_url"] = avatar_url

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [user["id"]]
        cur.execute(f"UPDATE users SET {set_clause} WHERE id = ?", values)
        conn.commit()

        # Re-fetch
        cur.execute("SELECT * FROM users WHERE id = ?", (user["id"],))
        user = dict(cur.fetchone())
    else:
        # Create new user
        cur.execute("""
            INSERT INTO users (provider, provider_id, email, name, avatar_url, created_at, last_login)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (provider, provider_id, email, name, avatar_url, now, now))
        conn.commit()

        user_id = cur.lastrowid
        cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        user = dict(cur.fetchone())
        logger.info("Created new user: id=%d, provider=%s, email=%s", user_id, provider, email)

    conn.close()
    return user


def get_user_by_id(db_path: str, user_id: int) -> Optional[Dict[str, Any]]:
    """Get a user by ID."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def update_user_app_version(db_path: str, user_id: int, app_version: str, patch_number: int = None):
    """Update user's app version and last_seen timestamp."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    cur.execute("""
        UPDATE users 
        SET app_version = ?, patch_number = ?, last_seen = ?
        WHERE id = ?
    """, (app_version, patch_number, now, user_id))
    conn.commit()
    conn.close()


def get_all_users_with_versions(db_path: str) -> list:
    """Get all users with their app version info."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT id, email, name, app_version, patch_number, last_seen, last_login, created_at
        FROM users
        ORDER BY last_seen DESC NULLS LAST
    """)
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# FastAPI Dependencies
# ---------------------------------------------------------------------------

_bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> Optional[Dict[str, Any]]:
    """
    Extract user from JWT token if present.
    Returns user dict or None (does NOT raise — allows fallback to API key auth).
    """
    if not credentials or not credentials.credentials:
        return None

    token = credentials.credentials

    # Skip if it looks like an API key (sk-...)
    if token.startswith("sk-"):
        return None

    payload = decode_token(token)
    if not payload or payload.get("type") != "access":
        return None

    user_id = int(payload["sub"])
    request.state.user_id = user_id
    request.state.user_email = payload.get("email")
    request.state.user_name = payload.get("name")

    return {
        "id": user_id,
        "email": payload.get("email"),
        "name": payload.get("name"),
    }


async def user_required(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> Dict[str, Any]:
    """
    Require a valid user JWT token. Raises 401 if not authenticated.
    """
    user = await get_current_user(request, credentials)
    if not user:
        raise HTTPException(
            status_code=401,
            detail={"code": "UNAUTHORIZED", "message": "Login required. Provide a valid JWT token."}
        )
    return user


# Admin whitelist - emails allowed to access admin endpoints
ADMIN_EMAILS = ["jon.suppe@gmail.com"]


async def get_admin_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> Dict[str, Any]:
    """
    Require admin user. Raises 401/403 if not authenticated or not admin.
    """
    user = await get_current_user(request, credentials)
    if not user:
        raise HTTPException(
            status_code=401,
            detail={"code": "UNAUTHORIZED", "message": "Login required."}
        )
    
    if user.get("email") not in ADMIN_EMAILS:
        raise HTTPException(
            status_code=403,
            detail={"code": "FORBIDDEN", "message": "Admin access required."}
        )
    
    return user


async def flexible_auth(
    request: Request,
    response: Response,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> Dict[str, Any]:
    """
    Accept either a JWT user token or an API key.
    Returns a dict with 'auth_type' ('user' or 'api_key') and relevant data.
    
    For user tokens: {'auth_type': 'user', 'user_id': int, 'email': str, 'name': str}
    For API keys: {'auth_type': 'api_key', 'key_data': dict}
    """
    from fastapi import Query as Q

    token = None
    if credentials and credentials.credentials:
        token = credentials.credentials

    # Check query param fallback
    if not token:
        token = request.query_params.get("api_key")

    if not token:
        raise HTTPException(
            status_code=401,
            detail={"code": "UNAUTHORIZED", "message": "Authentication required. Provide JWT token or API key."}
        )

    # API key path
    if token.startswith("sk-"):
        from .auth import lookup_key
        from .rate_limiter import rate_limiter as rl
        from .usage import usage_tracker as ut

        key_data = lookup_key(token)
        if not key_data:
            raise HTTPException(status_code=401, detail={"code": "UNAUTHORIZED", "message": "Invalid API key."})
        if not key_data.get("enabled", False):
            raise HTTPException(status_code=401, detail={"code": "UNAUTHORIZED", "message": "API key is disabled."})

        key_id = key_data["key_id"]
        rate_check = rl.check_rate_limit(key_data)
        if rate_check:
            raise HTTPException(status_code=429, detail={"code": "RATE_LIMITED", "message": rate_check["reason"]})

        rl.record_request(key_id)
        ut.record_request(key_id)

        headers = rl.get_rate_limit_headers(key_data)
        for name, value in headers.items():
            response.headers[name] = value

        request.state.key_data = key_data
        return {"auth_type": "api_key", "key_data": key_data, "user_id": None}

    # JWT token path
    payload = decode_token(token)
    if not payload or payload.get("type") != "access":
        raise HTTPException(
            status_code=401,
            detail={"code": "UNAUTHORIZED", "message": "Invalid or expired token."}
        )

    user_id = int(payload["sub"])
    request.state.user_id = user_id
    request.state.user_email = payload.get("email")
    request.state.user_name = payload.get("name")

    return {
        "auth_type": "user",
        "user_id": user_id,
        "email": payload.get("email"),
        "name": payload.get("name"),
        "key_data": None,
    }
