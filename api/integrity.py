"""
Play Integrity API verification for Android devices.
Prevents bot farms and emulators from abusing the API.
"""

import os
import json
import base64
import logging
import httpx
from typing import Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Cache for verified tokens (5 minute TTL)
_token_cache: dict[str, datetime] = {}
CACHE_TTL = timedelta(minutes=5)

# Skip integrity checks in these cases
BYPASS_USER_AGENTS = ["Mozilla", "PostmanRuntime", "curl"]  # Web/debug clients


def should_skip_integrity(user_agent: Optional[str], api_key: Optional[str]) -> bool:
    """Skip integrity for web clients and API key auth (admin)."""
    if api_key:
        return True  # API key users are trusted admins
    if user_agent:
        for bypass in BYPASS_USER_AGENTS:
            if bypass in user_agent:
                return True
    return False


async def verify_play_integrity(
    token: str,
    package_name: str = "com.speechscore.speechscore",
    project_number: Optional[str] = None
) -> Tuple[bool, str, dict]:
    """
    Verify a Play Integrity token with Google's servers.
    
    Returns:
        (is_valid, message, verdict_dict)
    """
    if not token:
        return False, "No integrity token provided", {}
    
    # Check cache first
    cache_key = token[:64]  # Use prefix as key
    if cache_key in _token_cache:
        if datetime.now() - _token_cache[cache_key] < CACHE_TTL:
            return True, "Token cached", {"cached": True}
        else:
            del _token_cache[cache_key]
    
    # Load credentials
    config_path = os.path.join(os.path.dirname(__file__), "integrity_config.json")
    if not os.path.exists(config_path):
        logger.warning("No integrity_config.json - integrity checks disabled")
        return True, "Integrity not configured", {"unconfigured": True}
    
    try:
        with open(config_path) as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load integrity config: {e}")
        return True, "Config error", {"error": str(e)}
    
    project_number = project_number or config.get("project_number")
    if not project_number:
        return True, "No project number configured", {"unconfigured": True}
    
    # Google Play Integrity API endpoint
    url = f"https://playintegrity.googleapis.com/v1/{project_number}:decodeIntegrityToken"
    
    try:
        async with httpx.AsyncClient() as client:
            # Use service account or API key
            headers = {"Content-Type": "application/json"}
            
            api_key = config.get("api_key")
            if api_key:
                url = f"{url}?key={api_key}"
            
            response = await client.post(
                url,
                json={"integrity_token": token},
                headers=headers,
                timeout=10.0
            )
            
            if response.status_code != 200:
                logger.warning(f"Integrity API error: {response.status_code} - {response.text}")
                return False, f"API error: {response.status_code}", {}
            
            data = response.json()
            verdict = data.get("tokenPayloadExternal", {})
            
            # Check device integrity
            device_integrity = verdict.get("deviceIntegrity", {})
            device_recognition = device_integrity.get("deviceRecognitionVerdict", [])
            
            # MEETS_DEVICE_INTEGRITY means genuine device (not emulator/rooted)
            is_genuine = "MEETS_DEVICE_INTEGRITY" in device_recognition
            
            # Check app integrity  
            app_integrity = verdict.get("appIntegrity", {})
            app_recognition = app_integrity.get("appRecognitionVerdict", "")
            is_valid_app = app_recognition in ["PLAY_RECOGNIZED", "UNRECOGNIZED_VERSION"]
            
            # Check package name
            request_details = verdict.get("requestDetails", {})
            received_package = request_details.get("requestPackageName", "")
            package_match = received_package == package_name
            
            is_valid = is_genuine and package_match
            
            if is_valid:
                _token_cache[cache_key] = datetime.now()
            
            return is_valid, "Verified" if is_valid else "Device integrity check failed", {
                "device_recognition": device_recognition,
                "app_recognition": app_recognition,
                "package_match": package_match,
                "is_genuine": is_genuine
            }
            
    except Exception as e:
        logger.error(f"Integrity verification failed: {e}")
        # Fail open for now - don't block users due to API issues
        return True, f"Verification error: {e}", {"error": str(e)}


def create_integrity_nonce(user_id: str, timestamp: Optional[int] = None) -> str:
    """
    Create a nonce for integrity requests.
    Should be unique per request and include user context.
    """
    import hashlib
    ts = timestamp or int(datetime.now().timestamp())
    data = f"{user_id}:{ts}:speechscore"
    return base64.urlsafe_b64encode(hashlib.sha256(data.encode()).digest()).decode()
