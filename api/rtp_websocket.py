"""
RTP-over-WebSocket for Low-Latency Speech-to-Text

Combines RTP's packet structure with WebSocket transport.
Works through Cloudflare/proxies while maintaining timing benefits.

Protocol:
  Client → Server:
    Binary WebSocket frames containing RTP packets
    JSON: {"type": "end"} to finalize
    
  Server → Client:
    JSON: {"type": "ready", "ssrc": 12345}
    JSON: {"type": "partial", "text": "hello wor"}
    JSON: {"type": "transcript", "text": "hello world", "is_final": true}
    JSON: {"type": "segment", "text": "...", "wpm": 145, "start": 0.0, "end": 1.5}
    JSON: {"type": "stats", ...}
"""

import asyncio
import struct
import json
import logging
import time
import threading
import queue
from typing import Optional, Dict, Any, Generator
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from collections import deque
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("speechscore.rtp_ws")

router = APIRouter(prefix="/ws", tags=["RTP WebSocket"])

# Constants
RTP_VERSION = 2
RTP_HEADER_SIZE = 12
SAMPLE_RATE = 16000
JITTER_BUFFER_MS = 30  # Smaller buffer for lower latency

# Filler words
FILLER_WORDS = {'um', 'uh', 'er', 'ah', 'like', 'you know', 'basically', 'actually'}
HEDGE_WORDS = {'i think', 'maybe', 'probably', 'sort of', 'kind of', 'i guess'}


@dataclass
class RTPPacket:
    """Parsed RTP packet"""
    sequence: int
    timestamp: int
    ssrc: int
    payload: bytes
    marker: bool = False
    received_at: float = field(default_factory=time.time)
    
    @classmethod
    def parse(cls, data: bytes) -> Optional['RTPPacket']:
        if len(data) < RTP_HEADER_SIZE:
            return None
        try:
            byte0, byte1, seq, ts, ssrc = struct.unpack('>BBHII', data[:12])
            version = (byte0 >> 6) & 0x03
            if version != RTP_VERSION:
                return None
            marker = bool((byte1 >> 7) & 0x01)
            payload = data[RTP_HEADER_SIZE:]
            return cls(sequence=seq, timestamp=ts, ssrc=ssrc, payload=payload, marker=marker)
        except:
            return None


class JitterBuffer:
    """Minimal jitter buffer for reordering packets"""
    
    def __init__(self, max_delay_ms: int = JITTER_BUFFER_MS):
        self.max_delay = max_delay_ms / 1000.0
        self.packets: Dict[int, RTPPacket] = {}
        self.next_seq: Optional[int] = None
        self.last_output_time: float = 0
        
    def add(self, packet: RTPPacket) -> Optional[bytes]:
        """Add packet, return audio if ready"""
        now = time.time()
        
        if self.next_seq is None:
            self.next_seq = packet.sequence
            self.last_output_time = now
            
        self.packets[packet.sequence] = packet
        
        # Output if we have next packet or waited too long
        if packet.sequence == self.next_seq or (now - self.last_output_time) > self.max_delay:
            return self._flush()
        return None
        
    def _flush(self) -> Optional[bytes]:
        """Flush ready packets"""
        if self.next_seq is None:
            return None
            
        chunks = []
        while self.next_seq in self.packets:
            pkt = self.packets.pop(self.next_seq)
            chunks.append(pkt.payload)
            self.next_seq = (self.next_seq + 1) & 0xFFFF
            
        self.last_output_time = time.time()
        return b''.join(chunks) if chunks else None
        
    def flush_all(self) -> Optional[bytes]:
        """Flush all remaining packets"""
        if not self.packets:
            return None
        chunks = [p.payload for p in sorted(self.packets.values(), key=lambda x: x.sequence)]
        self.packets.clear()
        return b''.join(chunks) if chunks else None


class RTPWebSocketSession:
    """Manages an RTP-over-WebSocket session"""
    
    def __init__(self, use_riva: bool = True):
        self.use_riva = use_riva
        self.jitter = JitterBuffer()
        self.ssrc: Optional[int] = None
        
        # Stats
        self.total_packets = 0
        self.total_bytes = 0
        self.total_audio_ms = 0
        self.start_time = time.time()
        
        # Transcript
        self.full_transcript = ""
        self.total_words = 0
        self.filler_count = 0
        self.hedge_count = 0
        
        # STT
        self._init_stt()
        
    def _init_stt(self):
        """Initialize STT engine"""
        if self.use_riva:
            try:
                import riva.client
                self.riva_client = riva.client
                self.riva_auth = riva.client.Auth(uri="localhost:50051")
                self.riva_asr = riva.client.ASRService(self.riva_auth)
                
                # Config for streaming with interim results
                self.riva_config = riva.client.RecognitionConfig(
                    encoding=riva.client.AudioEncoding.LINEAR_PCM,
                    sample_rate_hertz=SAMPLE_RATE,
                    language_code="en-US",
                    max_alternatives=1,
                    enable_automatic_punctuation=True,
                    verbatim_transcripts=False,
                )
                self.riva_streaming_config = riva.client.StreamingRecognitionConfig(
                    config=self.riva_config,
                    interim_results=True,  # Get partial results!
                )
                
                # Audio queue for streaming
                self.audio_queue = asyncio.Queue()
                self.streaming_task = None
                
                logger.info("RTP-WS: Using Riva STREAMING STT")
                return
            except Exception as e:
                logger.warning(f"Riva not available: {e}, falling back to Vosk")
                self.use_riva = False
        
        # Vosk fallback
        try:
            from vosk import Model, KaldiRecognizer
            model_path = "/home/melchior/vosk-models/vosk-model-en-us-0.22"
            self.vosk_model = Model(model_path)
            self.vosk_rec = KaldiRecognizer(self.vosk_model, SAMPLE_RATE)
            self.vosk_rec.SetWords(True)
            logger.info("RTP-WS: Using Vosk STT")
        except Exception as e:
            logger.error(f"No STT available: {e}")
            
    def process_packet(self, data: bytes) -> Optional[Dict[str, Any]]:
        """Process RTP packet, return transcript event if available"""
        packet = RTPPacket.parse(data)
        if not packet:
            return None
            
        if self.ssrc is None:
            self.ssrc = packet.ssrc
            
        self.total_packets += 1
        self.total_bytes += len(packet.payload)
        
        # Add to jitter buffer
        audio = self.jitter.add(packet)
        if audio:
            return self._transcribe(audio)
        return None
        
    def _transcribe(self, audio: bytes) -> Optional[Dict[str, Any]]:
        """Transcribe audio chunk"""
        audio_ms = len(audio) / (SAMPLE_RATE * 2) * 1000
        self.total_audio_ms += audio_ms
        
        if self.use_riva:
            return self._transcribe_riva(audio)
        else:
            return self._transcribe_vosk(audio)
            
    def _transcribe_riva(self, audio: bytes) -> Optional[Dict[str, Any]]:
        """Transcribe with Riva"""
        try:
            response = self.riva_asr.offline_recognize(audio, self.riva_config)
            if response.results and response.results[0].alternatives:
                text = response.results[0].alternatives[0].transcript.strip()
                if text:
                    return self._process_text(text, is_final=True)
        except Exception as e:
            logger.error(f"Riva error: {e}")
        return None
        
    def _transcribe_vosk(self, audio: bytes) -> Optional[Dict[str, Any]]:
        """Transcribe with Vosk"""
        try:
            if self.vosk_rec.AcceptWaveform(audio):
                result = json.loads(self.vosk_rec.Result())
                text = result.get("text", "").strip()
                if text:
                    return self._process_text(text, is_final=True)
            else:
                partial = json.loads(self.vosk_rec.PartialResult())
                text = partial.get("partial", "").strip()
                if text:
                    return self._process_text(text, is_final=False)
        except Exception as e:
            logger.error(f"Vosk error: {e}")
        return None
        
    def _process_text(self, text: str, is_final: bool) -> Dict[str, Any]:
        """Process transcribed text, detect fillers/hedges"""
        lower = text.lower()
        words = text.split()
        
        if is_final:
            self.full_transcript += " " + text
            self.total_words += len(words)
            
            # Count fillers
            for word in words:
                if word.lower() in FILLER_WORDS:
                    self.filler_count += 1
                    
            # Count hedges
            for hedge in HEDGE_WORDS:
                if hedge in lower:
                    self.hedge_count += 1
        
        # Calculate WPM (based on audio duration so far)
        duration_min = self.total_audio_ms / 60000
        wpm = int(self.total_words / duration_min) if duration_min > 0 else 0
        
        return {
            "type": "transcript" if is_final else "partial",
            "text": text,
            "is_final": is_final,
            "wpm": wpm,
            "total_words": self.total_words,
            "filler_count": self.filler_count,
        }
        
    def finalize(self) -> Optional[Dict[str, Any]]:
        """Finalize session, flush remaining audio"""
        audio = self.jitter.flush_all()
        result = None
        if audio:
            result = self._transcribe(audio)
            
        # Also get final result from Vosk
        if not self.use_riva and hasattr(self, 'vosk_rec'):
            try:
                final = json.loads(self.vosk_rec.FinalResult())
                text = final.get("text", "").strip()
                if text:
                    result = self._process_text(text, is_final=True)
            except:
                pass
                
        return result
        
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        duration = time.time() - self.start_time
        return {
            "type": "stats",
            "duration_sec": round(duration, 1),
            "audio_sec": round(self.total_audio_ms / 1000, 1),
            "total_packets": self.total_packets,
            "total_bytes": self.total_bytes,
            "total_words": self.total_words,
            "filler_count": self.filler_count,
            "hedge_count": self.hedge_count,
            "avg_wpm": int(self.total_words / (self.total_audio_ms / 60000)) if self.total_audio_ms > 0 else 0,
            "transcript": self.full_transcript.strip(),
        }


@router.websocket("/live-rtp")
async def rtp_websocket_stream(websocket: WebSocket):
    """
    RTP-over-WebSocket endpoint for low-latency streaming STT.
    
    Send RTP packets as binary WebSocket frames.
    Receive JSON transcripts.
    """
    await websocket.accept()
    logger.info("RTP-WS: Connected")
    
    session = RTPWebSocketSession(use_riva=True)
    
    await websocket.send_json({"type": "ready"})
    
    try:
        while True:
            message = await websocket.receive()
            
            if "bytes" in message:
                # RTP packet
                result = session.process_packet(message["bytes"])
                if result:
                    await websocket.send_json(result)
                    
            elif "text" in message:
                # Control message
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "end":
                        # Finalize
                        final = session.finalize()
                        if final:
                            await websocket.send_json(final)
                        stats = session.get_stats()
                        await websocket.send_json(stats)
                        break
                except json.JSONDecodeError:
                    pass
                    
    except WebSocketDisconnect:
        logger.info(f"RTP-WS: Disconnected (SSRC={session.ssrc})")
    except Exception as e:
        logger.error(f"RTP-WS error: {e}")
    finally:
        stats = session.get_stats()
        logger.info(f"RTP-WS stats: {stats['total_packets']} packets, {stats['total_words']} words, {stats['avg_wpm']} WPM")
