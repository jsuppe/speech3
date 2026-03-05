"""
API Versioning Utilities

Adds api_version to responses and provides version checking.
"""

from typing import Any, Dict, Optional
from datetime import datetime

# Current API version
API_VERSION = "1.0"
API_BUILD_DATE = "2026-03-03"

# Feature flags (can be checked by clients)
API_FEATURES = [
    "chunked_upload",
    "dementia_analysis",
    "pronunciation_analysis",
    "speaker_diarization",
    "presentation_coaching",
    "realtime_streaming",
]


def versioned_response(
    data: Any,
    deprecated: bool = False,
    deprecation_notice: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Wrap response data with API version info.
    
    Args:
        data: The response data (dict, list, or primitive)
        deprecated: Whether this endpoint is deprecated
        deprecation_notice: Message about deprecation timeline
        
    Returns:
        Response dict with api_version field
    """
    response = {
        "api_version": API_VERSION,
    }
    
    if deprecated:
        response["deprecated"] = True
        if deprecation_notice:
            response["deprecation_notice"] = deprecation_notice
    
    # Handle different data types
    if isinstance(data, dict):
        response.update(data)
    else:
        response["data"] = data
    
    return response


def version_info() -> Dict[str, Any]:
    """Get full version information."""
    return {
        "api_version": API_VERSION,
        "build_date": API_BUILD_DATE,
        "features": API_FEATURES,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


def check_client_version(client_version: Optional[str]) -> Dict[str, Any]:
    """
    Check if client version is compatible.
    
    Args:
        client_version: Client's expected API version (e.g., "1.0")
        
    Returns:
        Compatibility info dict
    """
    if not client_version:
        return {
            "compatible": True,
            "server_version": API_VERSION,
            "message": "No client version specified, assuming compatible",
        }
    
    try:
        client_major, client_minor = map(int, client_version.split(".")[:2])
        server_major, server_minor = map(int, API_VERSION.split(".")[:2])
        
        # Major version must match
        if client_major != server_major:
            return {
                "compatible": False,
                "server_version": API_VERSION,
                "client_version": client_version,
                "message": f"Major version mismatch. Client: {client_major}, Server: {server_major}. Please update the app.",
            }
        
        # Client minor can be <= server minor (server is backwards compatible)
        if client_minor > server_minor:
            return {
                "compatible": False,
                "server_version": API_VERSION,
                "client_version": client_version,
                "message": f"Client expects newer API features. Please update the server or downgrade the app.",
            }
        
        return {
            "compatible": True,
            "server_version": API_VERSION,
            "client_version": client_version,
        }
        
    except (ValueError, AttributeError):
        return {
            "compatible": True,
            "server_version": API_VERSION,
            "client_version": client_version,
            "message": "Could not parse client version, assuming compatible",
        }
