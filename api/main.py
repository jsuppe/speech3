"""
SpeechScore API — FastAPI entry point.
Loads ML models at startup, serves speech analysis endpoints.
"""

import os
import sys
import time
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
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

# Register frontend routes
try:
    from frontend.routes import router as frontend_router
    app.include_router(frontend_router, include_in_schema=False)
    logger.info("Frontend dashboard routes registered.")
except ImportError as e:
    logger.warning(f"Frontend not available: {e}")
