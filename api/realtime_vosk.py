"""
Real-time Streaming STT via Vosk

Ultra-low latency (~100-200ms) streaming speech-to-text.
Vosk processes audio incrementally, returning partial results as you speak.

Protocol:
  Client → Server:
    Binary PCM audio frames (16-bit, 16kHz, mono)
    {"type": "end"} to finalize
    
  Server → Client:
    {"type": "ready"}
    {"type": "partial", "text": "hello wor"}  # interim result
    {"type": "result", "text": "hello world", "confidence": 0.95}  # final result
    {"type": "stats", "total_words": 50, "duration": 30.5}
"""

import asyncio
import json
import logging
import os
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from vosk import Model, KaldiRecognizer, SetLogLevel

logger = logging.getLogger("speechscore.realtime_vosk")

router = APIRouter(prefix="/ws", tags=["realtime"])

# Suppress Vosk logging
SetLogLevel(-1)

# Global model (loaded once)
_vosk_model: Optional[Model] = None
_model_path = "/home/melchior/vosk-models/vosk-model-en-us-0.22"

def get_vosk_model() -> Model:
    """Get or load the Vosk model"""
    global _vosk_model
    if _vosk_model is None:
        if not os.path.exists(_model_path):
            raise RuntimeError(f"Vosk model not found at {_model_path}")
        logger.info(f"Loading Vosk model from {_model_path}")
        _vosk_model = Model(_model_path)
        logger.info("Vosk model loaded")
    return _vosk_model


# Filler words to detect
FILLER_WORDS = {'um', 'uh', 'er', 'ah', 'like', 'you know', 'so', 'basically',
                'actually', 'literally', 'right', 'okay', 'well'}


class VoskStreamSession:
    """Manages a streaming Vosk session"""
    
    def __init__(self, model: Model, sample_rate: int = 16000):
        self.recognizer = KaldiRecognizer(model, sample_rate)
        self.recognizer.SetWords(True)  # Enable word-level timestamps
        self.sample_rate = sample_rate
        
        # Stats
        self.total_words = 0
        self.total_audio_bytes = 0
        self.fillers = 0
        self.results = []
        
    def process_audio(self, audio_bytes: bytes) -> dict:
        """Process audio chunk, return result if available"""
        self.total_audio_bytes += len(audio_bytes)
        
        if self.recognizer.AcceptWaveform(audio_bytes):
            # Final result for this utterance
            result = json.loads(self.recognizer.Result())
            text = result.get("text", "")
            
            if text:
                self.total_words += len(text.split())
                self.results.append(result)
                
                # Count fillers
                for word in text.lower().split():
                    if word in FILLER_WORDS:
                        self.fillers += 1
                
                # Extract word timings if available
                words = result.get("result", [])
                return {
                    "type": "result",
                    "text": text,
                    "words": words,
                    "confidence": result.get("confidence", 0.9),
                }
        else:
            # Partial result
            partial = json.loads(self.recognizer.PartialResult())
            text = partial.get("partial", "")
            if text:
                return {
                    "type": "partial",
                    "text": text,
                }
        
        return None
    
    def finalize(self) -> dict:
        """Finalize and get final result"""
        result = json.loads(self.recognizer.FinalResult())
        text = result.get("text", "")
        
        if text:
            self.total_words += len(text.split())
            self.results.append(result)
        
        duration = self.total_audio_bytes / (self.sample_rate * 2)  # 16-bit = 2 bytes
        
        return {
            "type": "stats",
            "total_words": self.total_words,
            "fillers": self.fillers,
            "duration": round(duration, 1),
            "final_text": text,
        }


@router.websocket("/live-vosk")
async def live_vosk_stream(websocket: WebSocket):
    """
    WebSocket endpoint for low-latency Vosk streaming STT.
    
    Send raw PCM audio (16-bit, 16kHz, mono) as binary frames.
    Receive partial and final transcription results.
    """
    await websocket.accept()
    logger.info("🎤 Vosk streaming WebSocket connected")
    print("🎤 Vosk streaming WebSocket connected", flush=True)
    
    try:
        model = get_vosk_model()
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
        await websocket.close()
        return
    
    session = VoskStreamSession(model)
    await websocket.send_json({"type": "ready"})
    
    try:
        while True:
            message = await websocket.receive()
            
            if "text" in message:
                data = json.loads(message["text"])
                if data.get("type") == "end":
                    # Finalize and send stats
                    stats = session.finalize()
                    await websocket.send_json(stats)
                    break
            
            elif "bytes" in message:
                # Process audio
                audio_bytes = message["bytes"]
                result = session.process_audio(audio_bytes)
                
                if result:
                    await websocket.send_json(result)
    
    except WebSocketDisconnect:
        logger.info("Vosk streaming WebSocket disconnected")
        stats = session.finalize()
        logger.info(f"Session stats: {stats}")
    
    except Exception as e:
        logger.error(f"Vosk streaming error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
