"""
API Key authentication system for SpeechScore API.

Keys are stored as SHA-256 hashes in keys.json — plaintext keys are NEVER persisted.
Provides FastAPI dependencies for auth_required and admin_required.
"""

import hashlib
import json
import os
import secrets
import threading
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from fastapi import Request, Response, HTTPException, Query, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger("speechscore.auth")

# Path to keys file
KEYS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keys.json")

# In-memory key store: key_hash -> key_data
_keys: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()

# Valid tiers
VALID_TIERS = {"free", "pro", "enterprise", "admin"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_key(plaintext_key: str) -> str:
    """SHA-256 hash of a plaintext API key."""
    return hashlib.sha256(plaintext_key.encode("utf-8")).hexdigest()


def _generate_key() -> str:
    """Generate a new API key: sk-<32 hex chars>."""
    return f"sk-{secrets.token_hex(16)}"


def _generate_key_id() -> str:
    """Generate a short key_id (first 8 chars of a UUID4)."""
    return uuid.uuid4().hex[:8]


def _save_keys():
    """Persist keys to disk. Caller must hold _lock."""
    data = {"keys": list(_keys.values())}
    tmp_path = KEYS_FILE + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, KEYS_FILE)


def _load_keys():
    """Load keys from disk into memory."""
    global _keys
    if not os.path.exists(KEYS_FILE):
        return
    try:
        with open(KEYS_FILE, "r") as f:
            data = json.load(f)
        _keys = {}
        for entry in data.get("keys", []):
            key_hash = entry.get("key_hash")
            if key_hash:
                _keys[key_hash] = entry
        logger.info("Loaded %d API keys from %s", len(_keys), KEYS_FILE)
    except Exception as e:
        logger.error("Failed to load keys.json: %s", e)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_keys() -> Optional[str]:
    """
    Initialize the key system. If keys.json doesn't exist, create it with
    a default admin key. Returns the plaintext admin key if newly created,
    else None.
    """
    with _lock:
        _load_keys()

        if _keys:
            # Keys already exist
            return None

        # First run — create default admin key
        plaintext = _generate_key()
        key_id = _generate_key_id()
        key_hash = _hash_key(plaintext)

        entry = {
            "key_id": key_id,
            "name": "Default Admin",
            "key_hash": key_hash,
            "tier": "admin",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "enabled": True,
        }
        _keys[key_hash] = entry
        _save_keys()

        logger.info("Created default admin key (key_id=%s)", key_id)
        return plaintext


# ---------------------------------------------------------------------------
# Key lookup
# ---------------------------------------------------------------------------

def lookup_key(plaintext_key: str) -> Optional[Dict[str, Any]]:
    """Look up a key by its plaintext value. Returns key data or None."""
    key_hash = _hash_key(plaintext_key)
    with _lock:
        return _keys.get(key_hash)


def get_all_keys() -> list:
    """Return all key entries (for admin listing). Never includes key_hash."""
    with _lock:
        result = []
        for entry in _keys.values():
            safe = {k: v for k, v in entry.items() if k != "key_hash"}
            result.append(safe)
        return result


def get_key_by_id(key_id: str) -> Optional[Dict[str, Any]]:
    """Find a key entry by key_id."""
    with _lock:
        for entry in _keys.values():
            if entry["key_id"] == key_id:
                return entry
        return None


def create_key(name: str, tier: str) -> tuple:
    """
    Create a new API key. Returns (plaintext_key, key_entry).
    The plaintext key should be returned to the caller ONCE.
    """
    if tier not in VALID_TIERS:
        raise ValueError(f"Invalid tier: {tier}. Must be one of: {', '.join(sorted(VALID_TIERS))}")

    plaintext = _generate_key()
    key_id = _generate_key_id()
    key_hash = _hash_key(plaintext)

    entry = {
        "key_id": key_id,
        "name": name,
        "key_hash": key_hash,
        "tier": tier,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "enabled": True,
    }

    with _lock:
        _keys[key_hash] = entry
        _save_keys()

    logger.info("Created API key: key_id=%s, name=%s, tier=%s", key_id, name, tier)
    return plaintext, entry


def update_key(key_id: str, **updates) -> Optional[Dict[str, Any]]:
    """
    Update a key's fields (enabled, tier, name).
    Returns updated entry or None if not found.
    """
    with _lock:
        for key_hash, entry in _keys.items():
            if entry["key_id"] == key_id:
                if "enabled" in updates and isinstance(updates["enabled"], bool):
                    entry["enabled"] = updates["enabled"]
                if "tier" in updates and updates["tier"] in VALID_TIERS:
                    entry["tier"] = updates["tier"]
                if "name" in updates and isinstance(updates["name"], str):
                    entry["name"] = updates["name"]
                _save_keys()
                logger.info("Updated API key: key_id=%s, updates=%s", key_id, updates)
                return {k: v for k, v in entry.items() if k != "key_hash"}
        return None


def delete_key(key_id: str) -> bool:
    """Delete a key by key_id. Returns True if found and deleted."""
    with _lock:
        to_delete = None
        for key_hash, entry in _keys.items():
            if entry["key_id"] == key_id:
                to_delete = key_hash
                break
        if to_delete:
            del _keys[to_delete]
            _save_keys()
            logger.info("Deleted API key: key_id=%s", key_id)
            return True
        return False


# ---------------------------------------------------------------------------
# FastAPI Dependencies
# ---------------------------------------------------------------------------

_bearer_scheme = HTTPBearer(auto_error=False)


async def _extract_api_key(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    api_key: Optional[str] = Query(None, alias="api_key", include_in_schema=False),
) -> Optional[str]:
    """Extract API key from Authorization header or query param."""
    # Try Bearer token first
    if credentials and credentials.credentials:
        return credentials.credentials
    # Then query param
    if api_key:
        return api_key
    return None


async def auth_required(
    request: Request,
    response: Response,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    api_key: Optional[str] = Query(None, alias="api_key", include_in_schema=False),
) -> Dict[str, Any]:
    """
    FastAPI dependency: requires a valid, enabled API key.
    Also checks rate limits and records usage.
    Returns the key data dict and attaches it to request.state.
    """
    raw_key = None
    if credentials and credentials.credentials:
        raw_key = credentials.credentials
    elif api_key:
        raw_key = api_key

    if not raw_key:
        raise HTTPException(
            status_code=401,
            detail={"code": "UNAUTHORIZED", "message": "Missing API key. Provide via Authorization: Bearer <key> header or api_key query param."}
        )

    key_data = lookup_key(raw_key)
    if not key_data:
        raise HTTPException(
            status_code=401,
            detail={"code": "UNAUTHORIZED", "message": "Invalid API key."}
        )

    if not key_data.get("enabled", False):
        raise HTTPException(
            status_code=401,
            detail={"code": "UNAUTHORIZED", "message": "API key is disabled."}
        )

    # Check rate limits
    from .rate_limiter import rate_limiter
    rate_check = rate_limiter.check_rate_limit(key_data)
    if rate_check:
        # Add rate limit headers to response
        headers = rate_limiter.get_rate_limit_headers(key_data)
        raise HTTPException(
            status_code=429,
            detail={
                "code": "RATE_LIMITED",
                "message": rate_check["reason"],
                "retry_after_sec": rate_check["retry_after_sec"],
            },
            headers={
                "Retry-After": str(int(rate_check["retry_after_sec"])),
                **headers,
            },
        )

    # Record request in rate limiter and usage tracker
    key_id = key_data["key_id"]
    rate_limiter.record_request(key_id)

    from .usage import usage_tracker
    usage_tracker.record_request(key_id)

    # Add rate limit headers to response
    headers = rate_limiter.get_rate_limit_headers(key_data)
    for name, value in headers.items():
        response.headers[name] = value

    # Attach to request state for downstream use
    request.state.key_data = key_data
    return key_data


async def admin_required(
    key_data: Dict[str, Any] = Depends(auth_required),
) -> Dict[str, Any]:
    """
    FastAPI dependency: requires admin tier.
    """
    if key_data.get("tier") != "admin":
        raise HTTPException(
            status_code=403,
            detail={"code": "FORBIDDEN", "message": "Admin access required."}
        )
    return key_data
