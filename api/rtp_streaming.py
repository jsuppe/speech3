"""
True Streaming RTP-over-WebSocket with Riva

Uses Riva's streaming ASR for lowest possible latency.
Audio is fed continuously, partial results returned immediately.

Latency: ~100ms (vs ~300ms for chunked processing)
"""

import asyncio
import struct
import json
import logging
import time
import threading
import queue
import tempfile
import os
from typing import Optional, Dict, Any, Generator, List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("speechscore.rtp_stream")

# Whisper for high-quality segment transcription
_whisper_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="whisper")

router = APIRouter(prefix="/ws", tags=["RTP Streaming"])

SAMPLE_RATE = 16000
RTP_HEADER_SIZE = 12
FILLER_WORDS = {'um', 'uh', 'er', 'ah', 'like', 'you know', 'basically', 'actually'}

# Thread pool for Riva streaming
_executor = ThreadPoolExecutor(max_workers=4)


@dataclass
class RTPPacket:
    sequence: int
    timestamp: int
    ssrc: int
    payload: bytes
    
    @classmethod
    def parse(cls, data: bytes) -> Optional['RTPPacket']:
        if len(data) < RTP_HEADER_SIZE:
            return None
        try:
            _, _, seq, ts, ssrc = struct.unpack('>BBHII', data[:12])
            return cls(sequence=seq, timestamp=ts, ssrc=ssrc, payload=data[12:])
        except:
            return None


class RivaStreamingSession:
    """
    Hybrid streaming ASR: Riva (real-time) + Whisper (high-quality segments).
    
    Architecture:
    - Main thread: Receives RTP packets, adds to audio queue
    - Riva thread: Streams to Riva for instant results
    - Whisper thread: Processes segments for high-quality refinement
    - Result queue: Returns both instant and refined transcripts
    """
    
    def __init__(self):
        self.ssrc: Optional[int] = None
        self.start_time = time.time()
        
        # Audio input queue (thread-safe)
        self.audio_queue: queue.Queue = queue.Queue()
        
        # Result output queue (async)
        self.result_queue: asyncio.Queue = asyncio.Queue()
        
        # Stats
        self.total_packets = 0
        self.total_bytes = 0
        self.total_words = 0
        self.filler_count = 0
        self.full_transcript = ""
        
        # Streaming state
        self._running = False
        self._stream_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Segment tracking for Whisper refinement
        self._segment_id = 0
        self._current_segment_audio: List[bytes] = []
        self._current_segment_start = 0.0
        self._segment_riva_text = ""  # Riva's text for current segment
        
        # Pause detection
        self._last_speech_time: Optional[float] = None
        self._in_pause = False
        self._pause_start: Optional[float] = None
        self._pause_threshold_ms = 500  # Minimum pause to report (500ms)
        self._silence_threshold = 500  # RMS threshold for silence detection
        self._pending_whisper: Dict[int, bool] = {}  # Track pending Whisper jobs
        
        # Audio buffer for Whisper (raw PCM)
        self._whisper_audio_buffer: List[bytes] = []
        self._last_whisper_time = time.time()
        self._whisper_segment_seconds = 5.0  # Send to Whisper every 5 seconds
        
        # Pitch/tone tracking for real-time FFT visualization
        self._pitch_history: List[float] = []  # Last N pitch values
        self._pitch_history_max = 50  # Keep last 50 pitch readings
        self._last_pitch_send = time.time()
        self._pitch_send_interval = 0.1  # Send pitch data every 100ms
        self._segment_pitches: List[float] = []  # Pitches for current segment
        
        # Initialize Riva
        self._init_riva()
        
    def _init_riva(self):
        """Initialize Riva streaming client"""
        try:
            import riva.client
            self.riva = riva.client
            self.auth = riva.client.Auth(uri="localhost:50051")
            self.asr = riva.client.ASRService(self.auth)
            
            self.config = riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                sample_rate_hertz=SAMPLE_RATE,
                language_code="en-US",
                max_alternatives=1,
                enable_automatic_punctuation=True,
                verbatim_transcripts=True,  # Keep fillers and disfluencies!
            )
            self.streaming_config = riva.client.StreamingRecognitionConfig(
                config=self.config,
                interim_results=True,  # Key for low latency!
            )
            logger.info("Riva streaming initialized")
        except Exception as e:
            logger.error(f"Riva init failed: {e}")
            raise
            
    def start(self, loop: asyncio.AbstractEventLoop):
        """Start the streaming session"""
        self._running = True
        self._loop = loop
        self._stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self._stream_thread.start()
        logger.info("Riva streaming session started")
        
    def stop(self, save_audio: bool = True, user_id: int = None) -> str:
        """Stop the streaming session and optionally save audio.
        
        Args:
            save_audio: Whether to save audio as Opus
            user_id: User ID for audio organization
            
        Returns:
            Session ID if audio was saved, empty string otherwise
        """
        self._running = False
        self.audio_queue.put(None)  # Signal end
        if self._stream_thread:
            self._stream_thread.join(timeout=2)
        logger.info("Riva streaming session stopped")
        
        session_id = ""
        if save_audio and self._whisper_audio_buffer:
            try:
                from .audio_storage import save_session_audio
                
                # Combine all audio chunks
                all_audio = b''.join(self._whisper_audio_buffer)
                
                metadata = {
                    "duration_sec": time.time() - self.start_time,
                    "total_words": self.total_words,
                    "filler_count": self.filler_count,
                    "transcript": self.full_transcript.strip(),
                }
                
                session_id = save_session_audio(
                    pcm_data=all_audio,
                    user_id=user_id,
                    session_type="live",
                    metadata=metadata,
                )
                if session_id:
                    logger.info(f"Saved session audio: {session_id}")
            except Exception as e:
                logger.error(f"Failed to save session audio: {e}")
        
        return session_id or ""
        
    def add_packet(self, data: bytes):
        """Add RTP packet to processing queue"""
        packet = RTPPacket.parse(data)
        if not packet:
            return
            
        if self.ssrc is None:
            self.ssrc = packet.ssrc
            
        self.total_packets += 1
        self.total_bytes += len(packet.payload)
        
        # Add audio to queue for Riva streaming
        self.audio_queue.put(packet.payload)
        
        # Also buffer for Whisper segments
        self._whisper_audio_buffer.append(packet.payload)
        
        # Detect pauses based on audio energy
        self._detect_pause(packet.payload)
        
        # Extract pitch for real-time visualization
        self._process_pitch(packet.payload)
        
        # Check if we should send a segment to Whisper
        elapsed = time.time() - self._last_whisper_time
        if elapsed >= self._whisper_segment_seconds and self._whisper_audio_buffer:
            self._send_segment_to_whisper()
    
    def _detect_pause(self, audio_data: bytes):
        """Detect pauses in speech based on audio energy"""
        import struct
        
        # Calculate RMS energy of audio chunk
        samples = len(audio_data) // 2
        if samples == 0:
            return
            
        total = 0
        for i in range(0, len(audio_data) - 1, 2):
            sample = struct.unpack('<h', audio_data[i:i+2])[0]
            total += sample * sample
        rms = (total / samples) ** 0.5
        
        now = time.time()
        is_speech = rms > self._silence_threshold
        
        if is_speech:
            # Speech detected
            if self._in_pause and self._pause_start:
                # End of pause - calculate duration
                pause_duration_ms = (now - self._pause_start) * 1000
                if pause_duration_ms >= self._pause_threshold_ms:
                    # Report the pause
                    if self._loop:
                        asyncio.run_coroutine_threadsafe(
                            self.result_queue.put({
                                "type": "pause",
                                "duration_ms": int(pause_duration_ms),
                                "at": now - self.start_time,
                            }),
                            self._loop
                        )
            self._in_pause = False
            self._pause_start = None
            self._last_speech_time = now
        else:
            # Silence detected
            if not self._in_pause and self._last_speech_time:
                # Start of potential pause
                self._in_pause = True
                self._pause_start = now
    
    def _process_pitch(self, audio_data: bytes):
        """Extract pitch from audio chunk for real-time visualization"""
        import numpy as np
        
        # Convert bytes to samples
        samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        if len(samples) < 256:
            return
        
        # Simple autocorrelation-based pitch detection
        samples = samples - np.mean(samples)  # Remove DC offset
        
        # Normalize
        max_val = np.max(np.abs(samples))
        if max_val < 100:  # Too quiet, skip
            return
        samples = samples / max_val
        
        # Autocorrelation
        corr = np.correlate(samples, samples, mode='full')
        corr = corr[len(corr)//2:]
        
        # Find first peak after zero crossing
        d = np.diff(corr)
        starts = np.where((d[:-1] < 0) & (d[1:] >= 0))[0]
        
        if len(starts) > 0:
            # First positive peak
            peak_idx = starts[0]
            if peak_idx > 20:  # Min frequency ~800Hz at 16kHz sample rate
                pitch_hz = SAMPLE_RATE / peak_idx
                
                # Human voice range: 80-400 Hz
                if 80 <= pitch_hz <= 400:
                    self._pitch_history.append(pitch_hz)
                    self._segment_pitches.append(pitch_hz)
                    
                    # Keep history bounded
                    if len(self._pitch_history) > self._pitch_history_max:
                        self._pitch_history = self._pitch_history[-self._pitch_history_max:]
        
        # Send pitch data periodically
        now = time.time()
        if now - self._last_pitch_send >= self._pitch_send_interval and self._pitch_history:
            self._last_pitch_send = now
            
            # Calculate stats for the recent history
            recent = self._pitch_history[-20:] if len(self._pitch_history) >= 20 else self._pitch_history
            pitch_mean = np.mean(recent)
            pitch_std = np.std(recent)
            pitch_min = np.min(recent)
            pitch_max = np.max(recent)
            
            # Classify tone
            if pitch_std < 15:
                tone = "monotone"
            elif pitch_std < 30:
                tone = "steady"
            elif pitch_std < 50:
                tone = "varied"
            else:
                tone = "dynamic"
            
            # Send to client
            if self._loop:
                asyncio.run_coroutine_threadsafe(
                    self.result_queue.put({
                        "type": "pitch",
                        "pitch_hz": round(pitch_mean, 1),
                        "pitch_std": round(pitch_std, 1),
                        "pitch_min": round(pitch_min, 1),
                        "pitch_max": round(pitch_max, 1),
                        "tone": tone,
                        "history": [round(p, 1) for p in self._pitch_history[-30:]],  # Last 30 for graph
                    }),
                    self._loop
                )
        
    def _audio_generator(self) -> Generator[bytes, None, None]:
        """Generator that yields audio chunks from queue"""
        while self._running:
            try:
                audio = self.audio_queue.get(timeout=0.1)
                if audio is None:  # End signal
                    break
                yield audio
            except queue.Empty:
                continue
                
    def _stream_worker(self):
        """Background thread that runs Riva streaming"""
        try:
            responses = self.asr.streaming_response_generator(
                audio_chunks=self._audio_generator(),
                streaming_config=self.streaming_config,
            )
            
            for response in responses:
                if not self._running:
                    break
                    
                for result in response.results:
                    if not result.alternatives:
                        continue
                        
                    text = result.alternatives[0].transcript.strip()
                    if not text:
                        continue
                        
                    is_final = result.is_final
                    
                    # Process text
                    event = self._process_text(text, is_final)
                    
                    # Push to result queue (thread-safe via asyncio)
                    if self._loop:
                        asyncio.run_coroutine_threadsafe(
                            self.result_queue.put(event),
                            self._loop
                        )
                        
        except Exception as e:
            logger.error(f"Riva streaming error: {e}")
            if self._loop:
                asyncio.run_coroutine_threadsafe(
                    self.result_queue.put({"type": "error", "message": str(e)}),
                    self._loop
                )
                
    def _process_text(self, text: str, is_final: bool) -> Dict[str, Any]:
        """Process transcribed text from Riva"""
        import numpy as np
        
        words = text.split()
        
        # Calculate tone for this segment
        segment_tone = "neutral"
        segment_pitch_mean = 0.0
        segment_pitch_std = 0.0
        
        if self._segment_pitches:
            segment_pitch_mean = np.mean(self._segment_pitches)
            segment_pitch_std = np.std(self._segment_pitches)
            
            if segment_pitch_std < 15:
                segment_tone = "monotone"
            elif segment_pitch_std < 30:
                segment_tone = "steady"
            elif segment_pitch_std < 50:
                segment_tone = "varied"
            else:
                segment_tone = "dynamic"
        
        if is_final:
            self.full_transcript += " " + text
            self.total_words += len(words)
            self._segment_riva_text += " " + text  # Track for Whisper comparison
            
            for word in words:
                if word.lower() in FILLER_WORDS:
                    self.filler_count += 1
            
            # Clear segment pitches for next segment
            self._segment_pitches = []
        
        duration_min = (time.time() - self.start_time) / 60
        wpm = int(self.total_words / duration_min) if duration_min > 0 else 0
        
        return {
            "type": "transcript" if is_final else "partial",
            "text": text,
            "is_final": is_final,
            "wpm": wpm,
            "total_words": self.total_words,
            "filler_count": self.filler_count,
            "segment_id": self._segment_id,
            "tone": segment_tone,
            "pitch_mean": round(segment_pitch_mean, 1),
            "pitch_std": round(segment_pitch_std, 1),
        }
    
    def _send_segment_to_whisper(self):
        """Send accumulated audio to Whisper for high-quality transcription"""
        if not self._whisper_audio_buffer:
            return
            
        # Capture current segment data
        segment_id = self._segment_id
        audio_data = b''.join(self._whisper_audio_buffer)
        riva_text = self._segment_riva_text.strip()
        
        # Reset for next segment
        self._segment_id += 1
        self._whisper_audio_buffer = []
        self._segment_riva_text = ""
        self._last_whisper_time = time.time()
        self._pending_whisper[segment_id] = True
        
        # Submit to Whisper thread pool
        _whisper_executor.submit(
            self._process_whisper_segment,
            segment_id,
            audio_data,
            riva_text,
        )
        logger.debug(f"Sent segment {segment_id} to Whisper ({len(audio_data)} bytes)")
    
    def _process_whisper_segment(self, segment_id: int, audio_data: bytes, riva_text: str):
        """Process audio segment with Whisper (runs in thread pool)"""
        try:
            # Save audio to temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                # Write WAV header for 16kHz mono PCM
                import wave
                with wave.open(f.name, 'wb') as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)  # 16-bit
                    wav.setframerate(SAMPLE_RATE)
                    wav.writeframes(audio_data)
                temp_path = f.name
            
            # Transcribe with Whisper
            from faster_whisper import WhisperModel
            
            # Use the loaded model (or load if needed)
            model = WhisperModel("large-v3", device="cuda", compute_type="float16")
            segments, info = model.transcribe(temp_path, language="en", beam_size=5)
            
            whisper_text = " ".join([seg.text for seg in segments]).strip()
            
            # Clean up
            os.unlink(temp_path)
            
            # Send refined result back
            if whisper_text and self._loop:
                # Only send if different from Riva
                if whisper_text.lower() != riva_text.lower():
                    asyncio.run_coroutine_threadsafe(
                        self.result_queue.put({
                            "type": "refined",
                            "segment_id": segment_id,
                            "text": whisper_text,
                            "original": riva_text,
                        }),
                        self._loop
                    )
                    logger.debug(f"Segment {segment_id} refined: '{riva_text}' -> '{whisper_text}'")
                    
        except Exception as e:
            logger.error(f"Whisper segment error: {e}")
        finally:
            self._pending_whisper.pop(segment_id, None)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        duration = time.time() - self.start_time
        return {
            "type": "stats",
            "duration_sec": round(duration, 1),
            "total_packets": self.total_packets,
            "total_bytes": self.total_bytes,
            "total_words": self.total_words,
            "filler_count": self.filler_count,
            "avg_wpm": int(self.total_words / (duration / 60)) if duration > 0 else 0,
            "transcript": self.full_transcript.strip(),
        }


@router.websocket("/live-stream")
async def rtp_streaming_endpoint(websocket: WebSocket):
    """
    True streaming RTP-over-WebSocket endpoint.
    
    Lowest latency option - uses Riva's streaming ASR.
    Partial results returned as soon as available (~100ms).
    
    Protocol:
      Client → Server: Binary RTP packets, JSON {"type": "end"}
      Server → Client: JSON transcripts (partial + final)
    """
    await websocket.accept()
    logger.info("RTP streaming WebSocket connected")
    
    loop = asyncio.get_event_loop()
    
    try:
        session = RivaStreamingSession()
        session.start(loop)
    except Exception as e:
        await websocket.send_json({"type": "error", "message": f"Riva not available: {e}"})
        await websocket.close()
        return
    
    await websocket.send_json({"type": "ready"})
    
    async def send_results():
        """Task to send results from queue to WebSocket"""
        while True:
            try:
                event = await asyncio.wait_for(session.result_queue.get(), timeout=0.1)
                await websocket.send_json(event)
            except asyncio.TimeoutError:
                continue
            except Exception:
                break
    
    result_task = asyncio.create_task(send_results())
    
    try:
        while True:
            message = await websocket.receive()
            
            if "bytes" in message:
                session.add_packet(message["bytes"])
                
            elif "text" in message:
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "end":
                        break
                except:
                    pass
                    
    except WebSocketDisconnect:
        logger.info("RTP streaming WebSocket disconnected")
    except Exception as e:
        logger.error(f"RTP streaming error: {e}")
    finally:
        # Stop session and save audio as Opus
        session_id = session.stop(save_audio=True)
        result_task.cancel()
        
        stats = session.get_stats()
        if session_id:
            stats["session_id"] = session_id  # Include for replay
        try:
            await websocket.send_json(stats)
        except:
            pass
            
        logger.info(f"Streaming stats: {stats['total_packets']} pkts, {stats['total_words']} words, {stats['avg_wpm']} WPM")
