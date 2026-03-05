# SpeakFit API Compatibility Contract

## Overview

This document defines the API stability contract for SpeakFit backend services.
Multiple client apps (SpeakFit main, Memory, Present) share this API.

**Current API Version:** `1.0`

## Versioning

All responses include an `api_version` field:

```json
{
  "api_version": "1.0",
  "data": { ... }
}
```

### Version Guarantees

- **Major version (1.x → 2.x):** Breaking changes possible, clients must update
- **Minor version (1.0 → 1.1):** Additive only, fully backwards compatible

## Stability Tiers

### 🔒 STABLE — Do Not Break

These endpoints have locked signatures. Changes require major version bump.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/auth/google` | POST | Google Sign-In token exchange |
| `/v1/auth/refresh` | POST | Refresh JWT token |
| `/v1/auth/me` | GET | Get current user info |
| `/v1/speeches` | POST | Upload and analyze audio |
| `/v1/speeches/{id}` | GET | Get speech analysis results |
| `/v1/speeches/{id}` | PATCH | Update speech metadata |
| `/v1/speeches/{id}` | DELETE | Delete speech |
| `/v1/coach/chat` | POST | AI coach conversation |

### 🔓 EXTENSIBLE — Additive Changes Only

New optional parameters and response fields allowed. Existing fields frozen.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/speeches/{id}/pronunciation/*` | * | Pronunciation analysis |
| `/v1/dementia/*` | * | Dementia/cognitive analysis |
| `/v1/projects/*` | * | Project management |
| `/v1/speakers/*` | * | Speaker profiles |

### 🆕 NEW — Under Development

May change until promoted to EXTENSIBLE.

| Endpoint | Method | Description | Added |
|----------|--------|-------------|-------|
| `/v1/audio/chunked/*` | * | Chunked upload for long recordings | 2026-03-03 |
| `/v1/audio/sessions/*` | * | Session audio storage/replay | 2026-03-03 |

## Change Rules

### ✅ ALLOWED (non-breaking)

```python
# Add optional parameter with default
def analyze(audio, profile="default", new_param=None):

# Add new field to response
return {"score": 85, "new_field": "value"}

# Add new endpoint
@router.post("/v1/new-feature")

# Add new enum value
profile: Literal["default", "dementia", "presentation", "new_profile"]
```

### ❌ FORBIDDEN (breaking)

```python
# Remove or rename parameter
def analyze(audio, scoring_profile):  # was 'profile'

# Remove field from response
return {"score": 85}  # removed 'transcript'

# Change field type
return {"score": "85"}  # was int, now string

# Change endpoint path
@router.post("/v1/speech")  # was /v1/speeches

# Remove enum value
profile: Literal["default", "presentation"]  # removed 'dementia'
```

## Client Compatibility Matrix

| Client App | Min API Version | Uses Chunked Upload |
|------------|-----------------|---------------------|
| SpeakFit (main) | 1.0 | No |
| SpeakFit Memory | 1.0 | Yes |
| SpeakFit Present | 1.0 | No |
| Web Dashboard | 1.0 | No |

## Deprecation Process

1. Add `deprecated: true` to endpoint response
2. Log warning on server when deprecated endpoint called
3. Notify in changelog/release notes
4. Remove after 90 days (major version bump)

```json
{
  "api_version": "1.0",
  "deprecated": true,
  "deprecation_notice": "Use /v1/speeches instead. Removal: 2026-06-01",
  "data": { ... }
}
```

## Health & Version Endpoints

```
GET /health
→ {"status": "ok", "api_version": "1.0"}

GET /v1/version
→ {"api_version": "1.0", "build": "2026-03-03", "features": ["chunked_upload", "dementia"]}
```

## Changelog

### v1.0 (2026-03-03)
- Initial versioned API
- Added `api_version` to all responses
- Added `/v1/audio/chunked/*` endpoints
- Added `/v1/audio/sessions/*` endpoints
- Added `/v1/version` endpoint
