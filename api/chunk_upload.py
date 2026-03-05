"""
Chunked Audio Upload API

Supports reliable upload of long recordings via small chunks.
Designed for all-day recording scenarios (Medical/Memory apps).

Flow:
1. POST /sessions/start → session_id
2. POST /sessions/{id}/chunks (repeat, async)
3. POST /sessions/{id}/end → final session

Chunks can arrive out of order. Server reassembles on end.
"""

import os
import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass, field, asdict
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query, BackgroundTasks
from pydantic import BaseModel

from .audio_storage import AUDIO_STORAGE_DIR, encode_pcm_to_opus

logger = logging.getLogger("speechscore.chunk_upload")

router = APIRouter(prefix="/v1/audio/chunked", tags=["Chunked Upload"])

# Chunk storage directory
CHUNK_DIR = AUDIO_STORAGE_DIR / "chunks"
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

# Active sessions (in production, use Redis)
_active_sessions: Dict[str, "ChunkSession"] = {}


class StartSessionRequest(BaseModel):
    """Request to start a chunked upload session."""
    user_id: Optional[int] = None
    session_type: str = "memory"  # memory, medical, etc.
    sample_rate: int = 16000
    channels: int = 1
    chunk_format: str = "opus"  # opus or pcm
    metadata: Optional[dict] = None


class StartSessionResponse(BaseModel):
    """Response with session details."""
    session_id: str
    upload_url: str
    chunk_size_recommended_sec: int = 30
    max_chunk_size_bytes: int = 500_000  # 500KB


class ChunkUploadResponse(BaseModel):
    """Response after chunk upload."""
    session_id: str
    chunk_index: int
    received: bool
    chunks_received: int
    total_bytes: int


class EndSessionResponse(BaseModel):
    """Response after finalizing session."""
    session_id: str
    final_session_id: str
    total_chunks: int
    total_duration_sec: float
    total_bytes: int
    opus_file_size: int
    storage_path: str


@dataclass
class ChunkInfo:
    """Info about a single chunk."""
    index: int
    filename: str
    size_bytes: int
    received_at: str
    duration_sec: float = 0.0


@dataclass
class ChunkSession:
    """Active chunked upload session."""
    session_id: str
    user_id: Optional[int]
    session_type: str
    sample_rate: int
    channels: int
    chunk_format: str
    metadata: dict
    created_at: str
    chunks: Dict[int, ChunkInfo] = field(default_factory=dict)
    total_bytes: int = 0
    finalized: bool = False
    
    @property
    def chunk_dir(self) -> Path:
        return CHUNK_DIR / self.session_id
    
    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "session_type": self.session_type,
            "created_at": self.created_at,
            "chunks_received": len(self.chunks),
            "total_bytes": self.total_bytes,
            "finalized": self.finalized,
        }


def _generate_session_id() -> str:
    """Generate unique session ID."""
    import uuid
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"chunk_{timestamp}_{uuid.uuid4().hex[:8]}"


@router.post("/sessions/start", response_model=StartSessionResponse)
async def start_chunked_session(request: StartSessionRequest):
    """
    Start a new chunked upload session.
    
    Returns session_id and upload URL for chunks.
    Client should upload chunks to the provided URL.
    """
    session_id = _generate_session_id()
    
    session = ChunkSession(
        session_id=session_id,
        user_id=request.user_id,
        session_type=request.session_type,
        sample_rate=request.sample_rate,
        channels=request.channels,
        chunk_format=request.chunk_format,
        metadata=request.metadata or {},
        created_at=datetime.now().isoformat(),
    )
    
    # Create chunk directory
    session.chunk_dir.mkdir(parents=True, exist_ok=True)
    
    # Save session metadata
    meta_path = session.chunk_dir / "session.json"
    with open(meta_path, 'w') as f:
        json.dump(session.to_dict(), f, indent=2)
    
    _active_sessions[session_id] = session
    
    logger.info(f"Started chunked session: {session_id}")
    
    return StartSessionResponse(
        session_id=session_id,
        upload_url=f"/v1/audio/chunked/sessions/{session_id}/chunks",
        chunk_size_recommended_sec=30,
        max_chunk_size_bytes=500_000,
    )


@router.post("/sessions/{session_id}/chunks", response_model=ChunkUploadResponse)
async def upload_chunk(
    session_id: str,
    chunk_index: int = Form(..., description="Chunk sequence number (0-indexed)"),
    duration_sec: float = Form(0.0, description="Chunk duration in seconds"),
    chunk: UploadFile = File(..., description="Audio chunk (Opus or PCM)"),
):
    """
    Upload a single audio chunk.
    
    Chunks can arrive out of order. Server tracks by chunk_index.
    Client should retry failed uploads with same chunk_index.
    """
    # Get or restore session
    session = _active_sessions.get(session_id)
    if not session:
        # Try to restore from disk
        session = _restore_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        _active_sessions[session_id] = session
    
    if session.finalized:
        raise HTTPException(status_code=400, detail="Session already finalized")
    
    # Read chunk data
    chunk_data = await chunk.read()
    chunk_size = len(chunk_data)
    
    if chunk_size > 500_000:  # 500KB limit per chunk
        raise HTTPException(status_code=413, detail="Chunk too large (max 500KB)")
    
    # Save chunk to disk
    chunk_filename = f"chunk_{chunk_index:06d}.{session.chunk_format}"
    chunk_path = session.chunk_dir / chunk_filename
    
    with open(chunk_path, 'wb') as f:
        f.write(chunk_data)
    
    # Update session
    chunk_info = ChunkInfo(
        index=chunk_index,
        filename=chunk_filename,
        size_bytes=chunk_size,
        received_at=datetime.now().isoformat(),
        duration_sec=duration_sec,
    )
    session.chunks[chunk_index] = chunk_info
    session.total_bytes += chunk_size
    
    # Update session metadata on disk
    _save_session_meta(session)
    
    logger.info(f"Received chunk {chunk_index} for session {session_id}: {chunk_size} bytes")
    
    return ChunkUploadResponse(
        session_id=session_id,
        chunk_index=chunk_index,
        received=True,
        chunks_received=len(session.chunks),
        total_bytes=session.total_bytes,
    )


@router.post("/sessions/{session_id}/end", response_model=EndSessionResponse)
async def end_chunked_session(
    session_id: str,
    background_tasks: BackgroundTasks,
):
    """
    Finalize the chunked upload session.
    
    Assembles all chunks into a single Opus file.
    Returns final session details.
    """
    session = _active_sessions.get(session_id)
    if not session:
        session = _restore_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
    
    if session.finalized:
        raise HTTPException(status_code=400, detail="Session already finalized")
    
    if not session.chunks:
        raise HTTPException(status_code=400, detail="No chunks uploaded")
    
    # Assemble chunks in order
    sorted_indices = sorted(session.chunks.keys())
    
    # Check for gaps
    expected = list(range(sorted_indices[0], sorted_indices[-1] + 1))
    missing = set(expected) - set(sorted_indices)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing chunks: {sorted(missing)}. Upload missing chunks before finalizing."
        )
    
    # Assemble audio
    assembled_path, total_duration = await _assemble_chunks(session, sorted_indices)
    
    if not assembled_path:
        raise HTTPException(status_code=500, detail="Failed to assemble chunks")
    
    # Move to final storage location
    final_session_id = f"{session.session_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{session_id.split('_')[-1]}"
    
    if session.user_id:
        final_dir = AUDIO_STORAGE_DIR / f"user_{session.user_id}"
        final_dir.mkdir(exist_ok=True)
        final_path = final_dir / f"{final_session_id}.opus"
    else:
        final_path = AUDIO_STORAGE_DIR / f"{final_session_id}.opus"
    
    os.rename(assembled_path, final_path)
    opus_size = final_path.stat().st_size
    
    # Save metadata
    meta = {
        "session_id": final_session_id,
        "original_session_id": session_id,
        "user_id": session.user_id,
        "session_type": session.session_type,
        "created_at": session.created_at,
        "finalized_at": datetime.now().isoformat(),
        "total_chunks": len(session.chunks),
        "total_duration_sec": total_duration,
        "total_bytes_uploaded": session.total_bytes,
        "opus_file_size": opus_size,
        "compression_ratio": session.total_bytes / opus_size if opus_size > 0 else 0,
        "metadata": session.metadata,
    }
    
    meta_path = final_path.with_suffix('.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    # Mark session finalized
    session.finalized = True
    _save_session_meta(session)
    
    # Schedule cleanup of chunk files
    background_tasks.add_task(_cleanup_chunks, session)
    
    logger.info(f"Finalized session {session_id} → {final_session_id}: {opus_size} bytes Opus")
    
    return EndSessionResponse(
        session_id=session_id,
        final_session_id=final_session_id,
        total_chunks=len(session.chunks),
        total_duration_sec=total_duration,
        total_bytes=session.total_bytes,
        opus_file_size=opus_size,
        storage_path=str(final_path),
    )


@router.get("/sessions/{session_id}/status")
async def get_session_status(session_id: str):
    """
    Get current status of a chunked upload session.
    
    Useful for resuming interrupted uploads.
    """
    session = _active_sessions.get(session_id) or _restore_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    received_indices = sorted(session.chunks.keys())
    
    return {
        "session_id": session_id,
        "finalized": session.finalized,
        "chunks_received": len(session.chunks),
        "received_indices": received_indices,
        "total_bytes": session.total_bytes,
        "created_at": session.created_at,
        "next_expected_index": max(received_indices) + 1 if received_indices else 0,
    }


@router.delete("/sessions/{session_id}")
async def cancel_session(session_id: str):
    """
    Cancel and delete a chunked upload session.
    """
    session = _active_sessions.pop(session_id, None) or _restore_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Delete chunk directory
    import shutil
    if session.chunk_dir.exists():
        shutil.rmtree(session.chunk_dir)
    
    logger.info(f"Cancelled session {session_id}")
    
    return {"deleted": session_id}


def _restore_session(session_id: str) -> Optional[ChunkSession]:
    """Restore session from disk."""
    chunk_dir = CHUNK_DIR / session_id
    meta_path = chunk_dir / "session.json"
    
    if not meta_path.exists():
        return None
    
    try:
        with open(meta_path, 'r') as f:
            data = json.load(f)
        
        session = ChunkSession(
            session_id=data["session_id"],
            user_id=data.get("user_id"),
            session_type=data.get("session_type", "memory"),
            sample_rate=data.get("sample_rate", 16000),
            channels=data.get("channels", 1),
            chunk_format=data.get("chunk_format", "opus"),
            metadata=data.get("metadata", {}),
            created_at=data["created_at"],
            total_bytes=data.get("total_bytes", 0),
            finalized=data.get("finalized", False),
        )
        
        # Restore chunk info from files
        for chunk_file in chunk_dir.glob("chunk_*.opus"):
            idx = int(chunk_file.stem.split("_")[1])
            session.chunks[idx] = ChunkInfo(
                index=idx,
                filename=chunk_file.name,
                size_bytes=chunk_file.stat().st_size,
                received_at="restored",
            )
        for chunk_file in chunk_dir.glob("chunk_*.pcm"):
            idx = int(chunk_file.stem.split("_")[1])
            session.chunks[idx] = ChunkInfo(
                index=idx,
                filename=chunk_file.name,
                size_bytes=chunk_file.stat().st_size,
                received_at="restored",
            )
        
        return session
        
    except Exception as e:
        logger.error(f"Failed to restore session {session_id}: {e}")
        return None


def _save_session_meta(session: ChunkSession):
    """Save session metadata to disk."""
    meta_path = session.chunk_dir / "session.json"
    data = session.to_dict()
    data["chunks"] = {str(k): asdict(v) for k, v in session.chunks.items()}
    
    with open(meta_path, 'w') as f:
        json.dump(data, f, indent=2)


async def _assemble_chunks(session: ChunkSession, indices: List[int]) -> tuple:
    """Assemble chunks into single Opus file."""
    import subprocess
    import tempfile
    
    # Create file list for ffmpeg concat
    list_path = session.chunk_dir / "concat_list.txt"
    
    with open(list_path, 'w') as f:
        for idx in indices:
            chunk_info = session.chunks[idx]
            chunk_path = session.chunk_dir / chunk_info.filename
            f.write(f"file '{chunk_path}'\n")
    
    # Output path
    output_path = session.chunk_dir / "assembled.opus"
    
    try:
        if session.chunk_format == "opus":
            # Concat Opus files (copy codec, no re-encoding)
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(list_path),
                '-c', 'copy',
                str(output_path)
            ]
        else:
            # PCM chunks - need to concat and encode
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(list_path),
                '-f', 's16le',
                '-ar', str(session.sample_rate),
                '-ac', str(session.channels),
                '-c:a', 'libopus',
                '-b:a', '24k',
                '-vbr', 'on',
                '-application', 'voip',
                str(output_path)
            ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        
        if result.returncode != 0:
            logger.error(f"ffmpeg concat failed: {result.stderr.decode()}")
            return None, 0
        
        # Calculate total duration
        total_duration = sum(c.duration_sec for c in session.chunks.values())
        
        # If duration not provided, estimate from file size
        if total_duration == 0:
            # Estimate: ~3KB/sec at 24kbps Opus
            total_duration = session.total_bytes / 3000
        
        return str(output_path), total_duration
        
    except Exception as e:
        logger.error(f"Chunk assembly failed: {e}")
        return None, 0


def _cleanup_chunks(session: ChunkSession):
    """Clean up chunk files after finalization."""
    import shutil
    import time
    
    # Wait a bit in case client needs to retry
    time.sleep(60)
    
    try:
        if session.chunk_dir.exists():
            shutil.rmtree(session.chunk_dir)
            logger.info(f"Cleaned up chunks for session {session.session_id}")
    except Exception as e:
        logger.error(f"Chunk cleanup failed: {e}")
