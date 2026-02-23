"""
SpeechScore API — FastAPI entry point.
Loads ML models at startup, serves speech analysis endpoints.
"""

import os
import sys
import time
import asyncio
import logging
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.exceptions import RequestValidationError, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from . import config

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
os.makedirs(config.LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("speechscore")

# Suppress noisy library logs
for name in ["faster_whisper", "ctranslate2", "httpx", "urllib3", "transformers"]:
    logging.getLogger(name).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Lifespan — load models at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, start job manager, init auth & usage, cleanup on shutdown."""
    app.state.start_time = time.time()
    logger.info("SpeechScore API starting — loading models...")

    from . import pipeline_runner
    try:
        pipeline_runner.preload_models()
        logger.info("All models loaded. API ready.")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        # Still start the server so /health can report the problem

    # Initialize user auth system (users table, JWT, OAuth config)
    from .user_auth import init_user_auth, flexible_auth
    init_user_auth(config.DB_PATH)

    # Initialize API key system
    from .auth import init_keys
    admin_key = init_keys()
    if admin_key:
        logger.info("=" * 60)
        logger.info("FIRST RUN — Default admin API key created:")
        logger.info(f"  {admin_key}")
        logger.info("Store this key securely. It will NOT be shown again.")
        logger.info("=" * 60)
        # Also store it temporarily for retrieval
        app.state.first_run_admin_key = admin_key
    else:
        app.state.first_run_admin_key = None
        logger.info("API key system initialized (keys loaded from disk).")

    # Initialize usage tracker
    from .usage import usage_tracker
    usage_tracker.load()

    # Start the job manager (cleanup loop, etc.)
    from .job_manager import manager as job_manager
    loop = asyncio.get_running_loop()
    job_manager.start(loop)
    logger.info("Job manager started.")

    # Start usage flush loop
    usage_tracker.start_flush_loop(loop)

    yield

    # Shutdown
    job_manager.shutdown()
    usage_tracker.shutdown()
    logger.info("SpeechScore API shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SpeechScore API",
    description="""
# SpeechScore API

**Comprehensive speech analysis as a service.** Upload audio and get back transcription, text analysis, rhetorical analysis, sentiment, voice quality metrics, and more.

## Features

- **10 analysis modules:** transcription, text, advanced text, rhetorical, sentiment, audio, advanced audio, quality gate
- **Sync and async modes** — block for results or poll a job queue
- **Module selection** — run only what you need with `modules` or `preset` params
- **Quality gating** — automatically detect and flag noisy audio
- **Webhook support** — get notified when async jobs complete

## Authentication

All endpoints except `/v1/health` require an API key via `Authorization: Bearer sk-...` header or `api_key` query param.

## Quick Start

```bash
curl -X POST /v1/analyze -H "Authorization: Bearer sk-YOUR-KEY" -F "audio=@speech.mp3"
```
""",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Health", "description": "Server health and status"},
        {"name": "Analysis", "description": "Upload and analyze audio files"},
        {"name": "Jobs", "description": "Async job management and polling"},
        {"name": "Usage", "description": "Rate limits and usage tracking"},
        {"name": "Admin", "description": "API key management (admin only)"},
    ],
)

# CORS (allow all for dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Integrity middleware (logs device attestation status)
@app.middleware("http")
async def integrity_middleware(request: Request, call_next):
    """Check device integrity token from mobile apps."""
    from .integrity import should_skip_integrity, verify_play_integrity
    
    integrity_token = request.headers.get("X-Device-Integrity")
    user_agent = request.headers.get("User-Agent", "")
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
    
    # Skip for web clients and API key auth
    if should_skip_integrity(user_agent, api_key if api_key.startswith("sk-") else None):
        return await call_next(request)
    
    if integrity_token:
        # Debug bypass for development
        if integrity_token == "debug_bypass":
            request.state.integrity_status = "debug"
        else:
            is_valid, message, verdict = await verify_play_integrity(integrity_token)
            request.state.integrity_status = "valid" if is_valid else "invalid"
            if not is_valid:
                logger.warning(f"Device integrity check failed: {message}")
    else:
        request.state.integrity_status = "missing"
    
    return await call_next(request)


# HTTP exception handler (auth errors, etc.)
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Format HTTPException responses consistently."""
    detail = exc.detail
    if isinstance(detail, dict) and "code" in detail:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": detail},
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": "HTTP_ERROR", "message": str(detail)}},
    )


# Validation error handler (missing file, etc.)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    # Check if it's a missing 'audio' field
    for err in errors:
        if "audio" in str(err.get("loc", [])):
            return JSONResponse(
                status_code=400,
                content={"error": {"code": "MISSING_FILE", "message": "No audio file provided. Upload with field name 'audio'."}}
            )
    return JSONResponse(
        status_code=422,
        content={"error": {"code": "VALIDATION_ERROR", "message": str(errors)[:500]}}
    )


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "INTERNAL_ERROR", "message": str(exc)[:500]}}
    )


# Register API routes
from .router import router  # noqa: E402
app.include_router(router)

# Register authentication routes
from .auth_routes import auth_router  # noqa: E402
app.include_router(auth_router)

# Register pronunciation analysis routes
try:
    from .pronunciation_endpoints import router as pronunciation_router  # noqa: E402
    app.include_router(pronunciation_router)
    logger.info("Pronunciation analysis routes registered.")
except ImportError as e:
    logger.warning(f"Pronunciation endpoints not available: {e}")

# Register real-time pronunciation WebSocket
try:
    from .realtime_pronunciation import router as realtime_router  # noqa: E402
    app.include_router(realtime_router)
    logger.info("Real-time pronunciation WebSocket registered at /ws/pronunciation-stream")
except ImportError as e:
    logger.warning(f"Real-time pronunciation not available: {e}")

# Register gamification routes
try:
    from .gamification_endpoints import router as gamification_router  # noqa: E402
    app.include_router(gamification_router)
    logger.info("Gamification routes registered.")
except ImportError as e:
    logger.warning(f"Gamification endpoints not available: {e}")

# Register oratory analysis routes
try:
    from .oratory_endpoints import router as oratory_router  # noqa: E402
    app.include_router(oratory_router)
    logger.info("Oratory analysis routes registered.")
except ImportError as e:
    logger.warning(f"Oratory endpoints not available: {e}")

# Register presentation readiness routes
try:
    from .presentation_endpoints import router as presentation_router  # noqa: E402
    app.include_router(presentation_router)
    logger.info("Presentation readiness routes registered.")
except ImportError as e:
    logger.warning(f"Presentation endpoints not available: {e}")

# Register frontend routes
try:
    from frontend.routes import router as frontend_router
    app.include_router(frontend_router, include_in_schema=False)
    logger.info("Frontend dashboard routes registered.")
except ImportError as e:
    logger.warning(f"Frontend not available: {e}")

# Serve static files (images, etc.)
try:
    from fastapi.staticfiles import StaticFiles
    import os
    static_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")
        logger.info(f"Static files mounted at /static from {static_dir}")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")

# Serve Flutter web app at /flutter
try:
    from fastapi.staticfiles import StaticFiles
    flutter_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "static", "app")
    if os.path.exists(flutter_dir):
        app.mount("/flutter", StaticFiles(directory=flutter_dir, html=True), name="flutter")
        logger.info(f"Flutter web app mounted at /flutter from {flutter_dir}")
except Exception as e:
    logger.warning(f"Could not mount Flutter web app: {e}")

# Serve C23 WASM Compiler
try:
    from fastapi.staticfiles import StaticFiles
    c23_dir = "/home/melchior/dev/c23-wasm-compiler"
    if os.path.exists(c23_dir):
        app.mount("/dev/c23wasm", StaticFiles(directory=c23_dir, html=True), name="c23wasm")
        logger.info(f"C23 WASM compiler mounted at /dev/c23wasm")
except Exception as e:
    logger.warning(f"Could not mount C23 WASM compiler: {e}")


# ---------------------------------------------------------------------------
# Client Logs Endpoint
# ---------------------------------------------------------------------------

@app.post("/v1/logs/client")
async def receive_client_logs(data: dict):
    """
    Receive diagnostic logs from the mobile app.
    Stores logs for debugging purposes.
    """
    import json
    from datetime import datetime
    from pathlib import Path
    
    logs_dir = Path(__file__).parent.parent / "client_logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Add server timestamp
    data['server_received_at'] = datetime.utcnow().isoformat()
    
    # Write to log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    user_id = data.get('user_id', 'unknown')
    log_file = logs_dir / f"log_{timestamp}_{user_id}.json"
    
    with open(log_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    return {"status": "ok", "log_id": log_file.stem}

# ---------------------------------------------------------------------------
# User Feedback → Trello
# ---------------------------------------------------------------------------
@app.post("/v1/feedback")
async def submit_feedback(
    request: Request,
    feedback: str = Form(...),
    category: str = Form("general"),
    api_key: str = Form(None),
):
    """Submit user feedback - creates a Trello card."""
    import httpx
    from datetime import datetime
    from .user_auth import flexible_auth
    
    # Handle auth manually to avoid module-level dependency
    auth = {}
    try:
        # Try to get user info from JWT or API key
        from .user_auth import get_current_user, verify_api_key
        if api_key:
            key_data = verify_api_key(api_key)
            if key_data:
                auth = {"user_id": None, "email": "api_user", "key_data": key_data}
        else:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                user = await get_current_user(token)
                if user:
                    auth = {"user_id": user.id, "email": user.email}
    except Exception:
        pass
    
    user_id = auth.get("user_id")
    user_email = auth.get("email", "anonymous")
    
    # Trello API credentials
    TRELLO_KEY = "94dbc1af84f0abce181fd9433f69d727"
    TRELLO_TOKEN = "ATTA5e7ac0ff9877573186a0a8979f4b5525c0f3c68efdf77514ad640ac653de267aAF589D6A"
    BACKLOG_LIST_ID = "6980f79bb30707574404f57a"  # Backlog list
    
    # Create card
    card_name = f"📣 User Feedback: {feedback[:50]}..." if len(feedback) > 50 else f"📣 User Feedback: {feedback}"
    card_desc = f"""**User Feedback**

**Category:** {category}
**User ID:** {user_id}
**Email:** {user_email}
**Submitted:** {datetime.now().isoformat()}

---

{feedback}
"""
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.trello.com/1/cards",
                params={
                    "key": TRELLO_KEY,
                    "token": TRELLO_TOKEN,
                    "idList": BACKLOG_LIST_ID,
                    "name": card_name,
                    "desc": card_desc,
                }
            )
            response.raise_for_status()
            card_data = response.json()
            
            return {
                "success": True,
                "message": "Feedback submitted successfully",
                "card_url": card_data.get("shortUrl"),
            }
    except Exception as e:
        logger.exception("Failed to submit feedback to Trello")
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")
