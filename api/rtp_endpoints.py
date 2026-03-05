"""
RTP/UDP Streaming API Endpoints

Manages RTP server lifecycle and session registration.
"""

import asyncio
import logging
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from pydantic import BaseModel

from .rtp_receiver import (
    start_rtp_server, stop_rtp_server, get_rtp_server, get_stt_bridge,
    DEFAULT_PORT
)

logger = logging.getLogger("speechscore.rtp_endpoints")

router = APIRouter(prefix="/v1/rtp", tags=["RTP Streaming"])


class RTPSessionConfig(BaseModel):
    """Configuration for RTP session"""
    ssrc: int
    sample_rate: int = 16000
    encoding: str = "pcm16"  # pcm16, opus


class RTPServerStatus(BaseModel):
    """RTP server status"""
    running: bool
    port: int
    active_sessions: int
    total_packets: int = 0


# Server state
_server_started = False
_rtp_port = DEFAULT_PORT


@router.on_event("startup")
async def startup_rtp():
    """Start RTP server on API startup"""
    global _server_started
    try:
        await start_rtp_server(port=_rtp_port, use_riva=True)
        _server_started = True
        logger.info(f"RTP server started on port {_rtp_port}")
    except Exception as e:
        logger.error(f"Failed to start RTP server: {e}")


@router.get("/status")
async def get_status() -> dict:
    """Get RTP server status"""
    server = get_rtp_server()
    if not server:
        return {"running": False, "port": _rtp_port, "active_sessions": 0}
    
    return {
        "running": server.running,
        "port": _rtp_port,
        "active_sessions": len(server.sessions),
        "sessions": [
            {
                "ssrc": ssrc,
                "packets": s.total_packets,
                "bytes": s.total_bytes,
            }
            for ssrc, s in server.sessions.items()
        ]
    }


@router.post("/start")
async def start_server(port: int = Query(DEFAULT_PORT)) -> dict:
    """Start RTP server (if not running)"""
    global _server_started, _rtp_port
    
    if _server_started:
        return {"status": "already_running", "port": _rtp_port}
    
    try:
        _rtp_port = port
        await start_rtp_server(port=port, use_riva=True)
        _server_started = True
        return {"status": "started", "port": port}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.post("/stop")
async def stop_server() -> dict:
    """Stop RTP server"""
    global _server_started
    
    if not _server_started:
        return {"status": "not_running"}
    
    try:
        await stop_rtp_server()
        _server_started = False
        return {"status": "stopped"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.websocket("/transcript/{ssrc}")
async def transcript_stream(websocket: WebSocket, ssrc: int):
    """
    WebSocket for receiving transcripts from an RTP session.
    
    Client sends RTP audio via UDP, receives transcripts here.
    
    Protocol:
      Server → Client:
        {"type": "transcript", "text": "hello world", "is_final": true}
        {"type": "partial", "text": "hello wor"}
        {"type": "stats", "packets": 100, "duration": 10.5}
    """
    await websocket.accept()
    logger.info(f"Transcript WebSocket connected for SSRC {ssrc}")
    
    bridge = get_stt_bridge()
    if not bridge:
        await websocket.send_json({"type": "error", "message": "STT bridge not available"})
        await websocket.close()
        return
    
    # Queue for transcripts
    transcript_queue: asyncio.Queue = asyncio.Queue()
    
    def on_transcript(text: str, is_final: bool):
        """Callback when transcript is available"""
        try:
            transcript_queue.put_nowait({
                "type": "transcript" if is_final else "partial",
                "text": text.replace("[partial] ", ""),
                "is_final": is_final,
            })
        except asyncio.QueueFull:
            pass
    
    # Register callback
    bridge.register_callback(ssrc, on_transcript)
    
    try:
        await websocket.send_json({"type": "ready", "ssrc": ssrc})
        
        while True:
            try:
                # Wait for transcripts or client messages
                done, pending = await asyncio.wait(
                    [
                        asyncio.create_task(transcript_queue.get()),
                        asyncio.create_task(websocket.receive_json()),
                    ],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                
                for task in pending:
                    task.cancel()
                
                for task in done:
                    result = task.result()
                    
                    if isinstance(result, dict):
                        if "type" in result:
                            # Message from queue - send to client
                            await websocket.send_json(result)
                        elif result.get("action") == "end":
                            # Client ending session
                            server = get_rtp_server()
                            if server and ssrc in server.sessions:
                                session = server.sessions[ssrc]
                                await websocket.send_json({
                                    "type": "stats",
                                    "packets": session.total_packets,
                                    "bytes": session.total_bytes,
                                    "duration": session.total_bytes / (16000 * 2),
                                })
                            return
                            
            except asyncio.CancelledError:
                break
                
    except WebSocketDisconnect:
        logger.info(f"Transcript WebSocket disconnected for SSRC {ssrc}")
    except Exception as e:
        logger.error(f"Transcript WebSocket error: {e}")
    finally:
        bridge.unregister(ssrc)


@router.get("/info")
async def get_connection_info() -> dict:
    """
    Get RTP connection info for clients.
    
    Returns the server address and port for RTP streaming,
    plus the WebSocket URL for receiving transcripts.
    """
    return {
        "rtp": {
            "host": "api.speakfit.app",  # Client should use public hostname
            "port": _rtp_port,
            "protocol": "udp",
            "encoding": "pcm16",  # 16-bit linear PCM
            "sample_rate": 16000,
            "channels": 1,
        },
        "websocket": {
            "url": "wss://api.speakfit.app/v1/rtp/transcript/{ssrc}",
            "protocol": "json",
        },
        "notes": [
            "Generate a random SSRC (32-bit) for your session",
            "Send audio as RTP packets to host:port via UDP",
            "Connect to WebSocket with your SSRC to receive transcripts",
            "Use payload type 11 (L16) or 111 (Opus)",
        ]
    }
