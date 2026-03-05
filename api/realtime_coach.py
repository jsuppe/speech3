"""
Real-time Live Coach via WebSocket

Streams Opus audio from client, runs Whisper STT with VAD,
sends back word-level timestamps and analysis in real-time.

Architecture:
- Client sends: Opus audio frames continuously
- Server buffers, runs Whisper on speech segments (VAD-detected)
- Server sends: words with timestamps, segment WPM, fillers, hedges

Protocol:
  Client → Server:
    {"type": "config", "sample_rate": 16000}  # optional, once at start
    Binary Opus frames (prepended with 2-byte length)
    {"type": "end"}  # signals end of stream
    
  Server → Client:
    {"type": "ready"}
    {"type": "word", "text": "hello", "start": 1.23, "end": 1.45, "confidence": 0.98}
    {"type": "segment", "text": "hello world", "wpm": 142, "start": 1.0, "end": 2.5, "word_count": 2}
    {"type": "filler", "word": "um", "at": 2.31}
    {"type": "hedge", "phrase": "I think", "at": 3.0}
    {"type": "positive", "phrase": "definitely", "at": 4.5}
    {"type": "stats", "total_words": 50, "wpm_avg": 145, "fillers": 3, "hedges": 2}
    {"type": "error", "message": "..."}
"""

import asyncio
import json
import logging
import io
import struct
import numpy as np
import time
from collections import deque
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Deque
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger("speechscore.realtime_coach")

router = APIRouter(prefix="/ws", tags=["realtime"])

# Filler words to detect
FILLER_WORDS = {
    'um', 'uh', 'er', 'ah', 'like', 'you know', 'so', 'basically',
    'actually', 'literally', 'right', 'okay', 'well', 'i mean',
    'kind of', 'sort of',
}

# Positive indicators (confident language)
POSITIVE_INDICATORS = {
    'definitely', 'absolutely', 'clearly', 'specifically', 'precisely',
    'certainly', 'undoubtedly', 'exactly', 'for example', 'in fact',
    'importantly', 'significantly', 'notably', 'evidence shows',
    'data indicates', 'research confirms', 'studies show',
}

# Hedging phrases (weak language)
HEDGING_PHRASES = {
    'i think', 'i guess', 'i feel like', 'maybe', 'perhaps', 'probably',
    'sort of', 'kind of', 'a little bit', 'somewhat', 'might be',
    'could be', 'possibly', 'i believe', 'in my opinion', 'it seems',
    'i suppose', 'just', 'honestly', 'to be honest',
}


@dataclass
class WordInfo:
    """Word with timing information"""
    text: str
    start: float
    end: float
    confidence: float


@dataclass
class SegmentInfo:
    """A segment of speech (between pauses)"""
    words: List[WordInfo]
    start: float
    end: float
    
    @property
    def text(self) -> str:
        return ' '.join(w.text for w in self.words)
    
    @property
    def word_count(self) -> int:
        return len(self.words)
    
    @property
    def duration(self) -> float:
        return self.end - self.start
    
    @property
    def wpm(self) -> int:
        if self.duration > 0:
            return int(self.word_count * 60 / self.duration)
        return 0


class RealtimeCoachSession:
    """Manages a single real-time coaching session"""
    
    def __init__(self, whisper_model, sample_rate: int = 16000):
        self.whisper_model = whisper_model
        self.sample_rate = sample_rate
        
        # Audio buffer (float32 samples)
        self.audio_buffer = np.array([], dtype=np.float32)
        self.total_audio_duration = 0.0
        
        # Session stats
        self.total_words = 0
        self.total_fillers = 0
        self.total_hedges = 0
        self.total_positives = 0
        self.segments: List[SegmentInfo] = []
        
        # VAD settings
        self.silence_threshold = 0.015  # RMS threshold for silence
        self.min_speech_duration = 0.3  # Min speech segment (seconds)
        self.max_silence_duration = 0.8  # Max silence before segment break
        self.min_segment_duration = 0.5  # Min segment to process
        
        # State tracking
        self.is_speaking = False
        self.speech_start_sample = 0
        self.silence_samples = 0
        self.last_process_time = 0.0
        self.process_interval = 0.5  # Process every 500ms of speech
        
    def add_audio(self, audio_data: np.ndarray) -> List[dict]:
        """Add audio samples and return any detected events"""
        events = []
        
        # Append to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_data])
        self.total_audio_duration = len(self.audio_buffer) / self.sample_rate
        
        # Simple VAD: check RMS of recent audio
        window_samples = int(0.1 * self.sample_rate)  # 100ms window
        if len(self.audio_buffer) >= window_samples:
            recent = self.audio_buffer[-window_samples:]
            rms = np.sqrt(np.mean(recent ** 2))
            
            if rms > self.silence_threshold:
                # Speech detected
                if not self.is_speaking:
                    self.is_speaking = True
                    self.speech_start_sample = max(0, len(self.audio_buffer) - window_samples)
                self.silence_samples = 0
            else:
                # Silence
                self.silence_samples += len(audio_data)
                
                # Check if segment should end
                if self.is_speaking:
                    silence_duration = self.silence_samples / self.sample_rate
                    if silence_duration > self.max_silence_duration:
                        # Process the completed segment
                        segment_events = self._process_segment()
                        events.extend(segment_events)
                        self.is_speaking = False
        
        # Also process periodically during long speech
        if self.is_speaking:
            speech_samples = len(self.audio_buffer) - self.speech_start_sample
            speech_duration = speech_samples / self.sample_rate
            
            if speech_duration - self.last_process_time > self.process_interval:
                # Process current segment so far (streaming partial results)
                partial_events = self._process_partial()
                events.extend(partial_events)
                self.last_process_time = speech_duration
        
        return events
    
    def _process_segment(self) -> List[dict]:
        """Process a completed speech segment"""
        events = []
        
        # Extract segment audio
        end_sample = len(self.audio_buffer) - self.silence_samples
        segment_audio = self.audio_buffer[self.speech_start_sample:end_sample]
        
        duration = len(segment_audio) / self.sample_rate
        if duration < self.min_segment_duration:
            return events
        
        # Calculate timing offset
        offset = self.speech_start_sample / self.sample_rate
        
        # Run Whisper with word timestamps
        try:
            segments, info = self.whisper_model.transcribe(
                segment_audio,
                language="en",
                word_timestamps=True,
                vad_filter=False,  # We already did VAD
            )
            
            words = []
            for segment in segments:
                if hasattr(segment, 'words') and segment.words:
                    for word in segment.words:
                        word_info = WordInfo(
                            text=word.word.strip().lower(),
                            start=offset + word.start,
                            end=offset + word.end,
                            confidence=word.probability if hasattr(word, 'probability') else 0.9,
                        )
                        words.append(word_info)
                        
                        # Emit word event
                        events.append({
                            "type": "word",
                            "text": word_info.text,
                            "start": round(word_info.start, 2),
                            "end": round(word_info.end, 2),
                            "confidence": round(word_info.confidence, 2),
                        })
                        
                        # Check for fillers (case-insensitive, single words only here)
                        word_lower = word_info.text.lower()
                        if word_lower in FILLER_WORDS:
                            self.total_fillers += 1
                            events.append({
                                "type": "filler",
                                "word": word_info.text,
                                "at": round(word_info.start, 2),
                            })
                        
                        # Check for positive indicators (case-insensitive)
                        if word_lower in POSITIVE_INDICATORS:
                            self.total_positives += 1
                            events.append({
                                "type": "positive",
                                "phrase": word_info.text,
                                "at": round(word_info.start, 2),
                            })
            
            if words:
                # Create segment
                seg = SegmentInfo(
                    words=words,
                    start=words[0].start,
                    end=words[-1].end,
                )
                self.segments.append(seg)
                self.total_words += seg.word_count
                
                # Emit segment event
                events.append({
                    "type": "segment",
                    "text": seg.text,
                    "wpm": seg.wpm,
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "word_count": seg.word_count,
                })
                
                # Check for multi-word patterns (hedges, positives)
                text_lower = seg.text.lower()
                for hedge in HEDGING_PHRASES:
                    if hedge in text_lower:
                        self.total_hedges += 1
                        events.append({
                            "type": "hedge",
                            "phrase": hedge,
                            "at": round(seg.start, 2),
                        })
                
                for pos in POSITIVE_INDICATORS:
                    if ' ' in pos and pos in text_lower:
                        events.append({
                            "type": "positive",
                            "phrase": pos,
                            "at": round(seg.start, 2),
                        })
        
        except Exception as e:
            logger.error(f"Whisper error: {e}")
            events.append({"type": "error", "message": str(e)})
        
        # Reset for next segment
        self.last_process_time = 0.0
        
        return events
    
    def _process_partial(self) -> List[dict]:
        """Process partial segment for streaming feedback (minimal events)"""
        # For now, just return empty - full processing happens on segment end
        # Could add interim results here for even lower latency
        return []
    
    def get_stats(self) -> dict:
        """Get session statistics"""
        total_duration = sum(s.duration for s in self.segments)
        avg_wpm = int(self.total_words * 60 / total_duration) if total_duration > 0 else 0
        
        return {
            "type": "stats",
            "total_words": self.total_words,
            "wpm_avg": avg_wpm,
            "fillers": self.total_fillers,
            "hedges": self.total_hedges,
            "positives": self.total_positives,
            "duration": round(total_duration, 1),
        }
    
    def finalize(self) -> List[dict]:
        """Finalize session, process any remaining audio"""
        events = []
        
        # Process any remaining speech
        if self.is_speaking:
            self.silence_samples = int(self.max_silence_duration * self.sample_rate) + 1
            events.extend(self._process_segment())
        
        # Add final stats
        events.append(self.get_stats())
        
        return events


def decode_opus_to_pcm(opus_data: bytes, sample_rate: int = 16000) -> np.ndarray:
    """Decode Opus audio to PCM float32 samples"""
    try:
        import opuslib
        
        # Create decoder (mono, 16kHz)
        decoder = opuslib.Decoder(sample_rate, 1)
        
        # Decode (frame size = 20ms = 320 samples at 16kHz)
        frame_size = int(sample_rate * 0.02)
        pcm = decoder.decode(opus_data, frame_size)
        
        # Convert to float32
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        return samples
        
    except ImportError:
        # Fallback: assume raw PCM if opuslib not available
        logger.warning("opuslib not installed, assuming raw PCM")
        samples = np.frombuffer(opus_data, dtype=np.int16).astype(np.float32) / 32768.0
        return samples


def _get_whisper_model():
    """Get the Whisper model from pipeline_runner"""
    from . import pipeline_runner
    if pipeline_runner._whisper_model is None:
        pipeline_runner.load_models()
    return pipeline_runner._whisper_model


@router.websocket("/live-coach")
async def live_coach_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time live coaching.
    
    Client sends Opus-encoded audio frames, server returns
    word-level transcription with timing and analysis.
    """
    await websocket.accept()
    logger.info("Live coach WebSocket connected")
    
    # Get Whisper model from pipeline_runner
    whisper_model = _get_whisper_model()
    if whisper_model is None:
        await websocket.send_json({"type": "error", "message": "Whisper model not loaded"})
        await websocket.close()
        return
    
    session = RealtimeCoachSession(whisper_model)
    await websocket.send_json({"type": "ready"})
    
    try:
        while True:
            message = await websocket.receive()
            
            if "text" in message:
                # JSON message
                data = json.loads(message["text"])
                msg_type = data.get("type")
                
                if msg_type == "config":
                    # Update session config
                    if "sample_rate" in data:
                        session.sample_rate = data["sample_rate"]
                    await websocket.send_json({"type": "configured", "sample_rate": session.sample_rate})
                
                elif msg_type == "end":
                    # Finalize and close
                    events = session.finalize()
                    for event in events:
                        await websocket.send_json(event)
                    break
                
                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
            
            elif "bytes" in message:
                # Binary audio data
                audio_bytes = message["bytes"]
                
                # Decode Opus to PCM
                try:
                    samples = decode_opus_to_pcm(audio_bytes, session.sample_rate)
                except Exception as e:
                    logger.error(f"Audio decode error: {e}")
                    continue
                
                # Process and get events
                events = session.add_audio(samples)
                
                # Send events to client
                for event in events:
                    await websocket.send_json(event)
    
    except WebSocketDisconnect:
        logger.info("Live coach WebSocket disconnected")
        # Finalize session
        events = session.finalize()
        logger.info(f"Session stats: {events[-1] if events else 'none'}")
    
    except Exception as e:
        logger.error(f"Live coach error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass


# Alternative endpoint that accepts raw PCM (simpler, no Opus dependency)
@router.websocket("/live-coach-pcm")
async def live_coach_pcm_stream(websocket: WebSocket):
    """
    WebSocket endpoint for raw PCM audio (16-bit, 16kHz, mono).
    Simpler protocol - just send raw audio bytes.
    """
    await websocket.accept()
    logger.info("Live coach PCM WebSocket connected")
    
    # Get Whisper model from pipeline_runner
    whisper_model = _get_whisper_model()
    if whisper_model is None:
        await websocket.send_json({"type": "error", "message": "Whisper model not loaded"})
        await websocket.close()
        return
    
    session = RealtimeCoachSession(whisper_model)
    await websocket.send_json({"type": "ready"})
    
    try:
        while True:
            message = await websocket.receive()
            
            if "text" in message:
                data = json.loads(message["text"])
                if data.get("type") == "end":
                    events = session.finalize()
                    for event in events:
                        await websocket.send_json(event)
                    break
            
            elif "bytes" in message:
                # Raw PCM 16-bit samples
                audio_bytes = message["bytes"]
                samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                
                events = session.add_audio(samples)
                for event in events:
                    await websocket.send_json(event)
    
    except WebSocketDisconnect:
        logger.info("Live coach PCM WebSocket disconnected")
        events = session.finalize()
        logger.info(f"Session stats: {events[-1] if events else 'none'}")
    
    except Exception as e:
        logger.error(f"Live coach PCM error: {e}")
