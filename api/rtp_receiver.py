"""
RTP/UDP Audio Receiver for Low-Latency Speech-to-Text

Receives RTP packets over UDP, reassembles audio stream,
and feeds to Riva/Vosk for real-time transcription.

Protocol:
- Client sends RTP packets (RFC 3550) over UDP
- Server reassembles and processes with STT
- Transcripts sent back via WebSocket or HTTP polling

RTP Packet Format:
  0                   1                   2                   3
  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |V=2|P|X|  CC   |M|     PT      |       sequence number         |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                           timestamp                           |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |           synchronization source (SSRC) identifier            |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                          payload...                           |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
"""

import asyncio
import struct
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Any
from collections import deque
import threading

logger = logging.getLogger("speechscore.rtp")

# RTP Constants
RTP_VERSION = 2
RTP_HEADER_SIZE = 12
PAYLOAD_TYPE_PCMU = 0    # G.711 μ-law
PAYLOAD_TYPE_PCMA = 8    # G.711 A-law  
PAYLOAD_TYPE_L16 = 11    # Linear 16-bit PCM
PAYLOAD_TYPE_OPUS = 111  # Opus (dynamic)

# Server config
DEFAULT_PORT = 5004
JITTER_BUFFER_MS = 50  # Jitter buffer size
MAX_PACKET_SIZE = 1500
SAMPLE_RATE = 16000


@dataclass
class RTPPacket:
    """Parsed RTP packet"""
    version: int
    padding: bool
    extension: bool
    cc: int  # CSRC count
    marker: bool
    payload_type: int
    sequence: int
    timestamp: int
    ssrc: int
    payload: bytes
    received_at: float = field(default_factory=time.time)
    
    @classmethod
    def parse(cls, data: bytes) -> Optional['RTPPacket']:
        """Parse RTP packet from raw bytes"""
        if len(data) < RTP_HEADER_SIZE:
            return None
        
        try:
            # Parse header
            byte0, byte1, seq, ts, ssrc = struct.unpack('>BBHII', data[:12])
            
            version = (byte0 >> 6) & 0x03
            if version != RTP_VERSION:
                return None
            
            padding = bool((byte0 >> 5) & 0x01)
            extension = bool((byte0 >> 4) & 0x01)
            cc = byte0 & 0x0F
            marker = bool((byte1 >> 7) & 0x01)
            payload_type = byte1 & 0x7F
            
            # Skip CSRC list if present
            header_size = RTP_HEADER_SIZE + (cc * 4)
            
            # Handle extension header
            if extension and len(data) >= header_size + 4:
                ext_len = struct.unpack('>H', data[header_size+2:header_size+4])[0]
                header_size += 4 + (ext_len * 4)
            
            # Handle padding
            payload_end = len(data)
            if padding:
                padding_len = data[-1]
                payload_end -= padding_len
            
            payload = data[header_size:payload_end]
            
            return cls(
                version=version,
                padding=padding,
                extension=extension,
                cc=cc,
                marker=marker,
                payload_type=payload_type,
                sequence=seq,
                timestamp=ts,
                ssrc=ssrc,
                payload=payload,
            )
        except Exception as e:
            logger.warning(f"Failed to parse RTP packet: {e}")
            return None


class JitterBuffer:
    """
    Jitter buffer for reordering RTP packets and smoothing playback.
    Handles out-of-order packets and gaps.
    """
    
    def __init__(self, buffer_ms: int = JITTER_BUFFER_MS, sample_rate: int = SAMPLE_RATE):
        self.buffer_ms = buffer_ms
        self.sample_rate = sample_rate
        self.packets: Dict[int, RTPPacket] = {}
        self.next_seq: Optional[int] = None
        self.lock = threading.Lock()
        
    def add(self, packet: RTPPacket) -> None:
        """Add packet to buffer"""
        with self.lock:
            self.packets[packet.sequence] = packet
            
            if self.next_seq is None:
                self.next_seq = packet.sequence
    
    def get_ready_audio(self) -> Optional[bytes]:
        """
        Get audio that's ready for processing.
        Returns contiguous audio bytes, or None if buffer not ready.
        """
        with self.lock:
            if self.next_seq is None or not self.packets:
                return None
            
            # Collect contiguous packets
            audio_chunks = []
            current_seq = self.next_seq
            max_gap = 3  # Allow small gaps
            
            while True:
                if current_seq in self.packets:
                    pkt = self.packets.pop(current_seq)
                    audio_chunks.append(pkt.payload)
                    current_seq = (current_seq + 1) & 0xFFFF
                elif len(audio_chunks) > 0:
                    # Check if we should wait for more or proceed
                    gap_packets = 0
                    for i in range(1, max_gap + 1):
                        check_seq = (current_seq + i) & 0xFFFF
                        if check_seq in self.packets:
                            # Insert silence for gap
                            silence_samples = 320  # 20ms at 16kHz
                            audio_chunks.append(bytes(silence_samples * 2))
                            current_seq = (current_seq + 1) & 0xFFFF
                            gap_packets += 1
                            break
                    
                    if gap_packets == 0:
                        break
                else:
                    break
            
            self.next_seq = current_seq
            
            if audio_chunks:
                return b''.join(audio_chunks)
            return None


@dataclass
class RTPSession:
    """Manages an RTP streaming session"""
    ssrc: int
    created_at: float = field(default_factory=time.time)
    last_packet_at: float = field(default_factory=time.time)
    jitter_buffer: JitterBuffer = field(default_factory=JitterBuffer)
    total_packets: int = 0
    total_bytes: int = 0
    audio_callback: Optional[Callable[[bytes], Any]] = None
    
    def add_packet(self, packet: RTPPacket) -> Optional[bytes]:
        """Add packet and return audio if ready"""
        self.last_packet_at = time.time()
        self.total_packets += 1
        self.total_bytes += len(packet.payload)
        
        self.jitter_buffer.add(packet)
        return self.jitter_buffer.get_ready_audio()


class RTPServer:
    """
    Async UDP server for receiving RTP audio streams.
    Manages multiple concurrent sessions by SSRC.
    """
    
    def __init__(
        self, 
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        on_audio: Optional[Callable[[int, bytes], Any]] = None,
    ):
        self.host = host
        self.port = port
        self.on_audio = on_audio  # Callback: (ssrc, audio_bytes)
        self.sessions: Dict[int, RTPSession] = {}
        self.transport: Optional[asyncio.DatagramTransport] = None
        self.running = False
        self._cleanup_task: Optional[asyncio.Task] = None
        
    async def start(self) -> None:
        """Start the RTP server"""
        loop = asyncio.get_event_loop()
        
        self.transport, _ = await loop.create_datagram_endpoint(
            lambda: RTPProtocol(self),
            local_addr=(self.host, self.port),
        )
        
        self.running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_sessions())
        logger.info(f"RTP server listening on {self.host}:{self.port}")
        
    async def stop(self) -> None:
        """Stop the RTP server"""
        self.running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            
        if self.transport:
            self.transport.close()
            
        logger.info("RTP server stopped")
        
    async def _cleanup_sessions(self) -> None:
        """Clean up stale sessions"""
        while self.running:
            await asyncio.sleep(30)
            
            now = time.time()
            stale = [
                ssrc for ssrc, session in self.sessions.items()
                if now - session.last_packet_at > 60  # 60s timeout
            ]
            
            for ssrc in stale:
                logger.info(f"Cleaning up stale session {ssrc}")
                del self.sessions[ssrc]
                
    def handle_packet(self, data: bytes, addr: tuple) -> None:
        """Handle incoming RTP packet"""
        packet = RTPPacket.parse(data)
        if not packet:
            return
        
        # Get or create session
        if packet.ssrc not in self.sessions:
            self.sessions[packet.ssrc] = RTPSession(ssrc=packet.ssrc)
            logger.info(f"New RTP session from {addr}, SSRC={packet.ssrc}")
        
        session = self.sessions[packet.ssrc]
        audio = session.add_packet(packet)
        
        if audio and self.on_audio:
            # Process audio asynchronously
            asyncio.create_task(self._process_audio(packet.ssrc, audio))
            
    async def _process_audio(self, ssrc: int, audio: bytes) -> None:
        """Process audio through STT"""
        try:
            if self.on_audio:
                result = self.on_audio(ssrc, audio)
                if asyncio.iscoroutine(result):
                    await result
        except Exception as e:
            logger.error(f"Audio processing error: {e}")


class RTPProtocol(asyncio.DatagramProtocol):
    """Asyncio protocol for RTP/UDP"""
    
    def __init__(self, server: RTPServer):
        self.server = server
        
    def datagram_received(self, data: bytes, addr: tuple) -> None:
        self.server.handle_packet(data, addr)
        
    def error_received(self, exc: Exception) -> None:
        logger.error(f"RTP protocol error: {exc}")


# ============ STT Integration ============

class RTPSttBridge:
    """
    Bridge between RTP receiver and STT engine (Riva/Vosk).
    Manages transcription sessions per SSRC.
    """
    
    def __init__(self, use_riva: bool = True):
        self.use_riva = use_riva
        self.transcripts: Dict[int, str] = {}
        self.callbacks: Dict[int, Callable[[str, bool], Any]] = {}
        
        # Initialize STT engine
        if use_riva:
            self._init_riva()
        else:
            self._init_vosk()
            
    def _init_riva(self) -> None:
        """Initialize Riva ASR"""
        try:
            import riva.client
            self.riva_auth = riva.client.Auth(uri="localhost:50051")
            self.riva_asr = riva.client.ASRService(self.riva_auth)
            self.riva_config = riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                sample_rate_hertz=SAMPLE_RATE,
                language_code="en-US",
                max_alternatives=1,
                enable_automatic_punctuation=True,
            )
            logger.info("Riva ASR initialized for RTP bridge")
        except Exception as e:
            logger.error(f"Failed to init Riva: {e}")
            self.use_riva = False
            self._init_vosk()
            
    def _init_vosk(self) -> None:
        """Initialize Vosk ASR"""
        try:
            from vosk import Model, KaldiRecognizer
            model_path = "/home/melchior/vosk-models/vosk-model-en-us-0.22"
            self.vosk_model = Model(model_path)
            self.vosk_recognizers: Dict[int, Any] = {}
            logger.info("Vosk ASR initialized for RTP bridge")
        except Exception as e:
            logger.error(f"Failed to init Vosk: {e}")
    
    def process_audio(self, ssrc: int, audio: bytes) -> Optional[str]:
        """Process audio and return transcript if available"""
        if self.use_riva:
            return self._process_riva(ssrc, audio)
        else:
            return self._process_vosk(ssrc, audio)
            
    def _process_riva(self, ssrc: int, audio: bytes) -> Optional[str]:
        """Process with Riva"""
        try:
            response = self.riva_asr.offline_recognize(audio, self.riva_config)
            if response.results:
                text = response.results[0].alternatives[0].transcript
                return text.strip()
        except Exception as e:
            logger.error(f"Riva processing error: {e}")
        return None
        
    def _process_vosk(self, ssrc: int, audio: bytes) -> Optional[str]:
        """Process with Vosk"""
        try:
            from vosk import KaldiRecognizer
            import json
            
            if ssrc not in self.vosk_recognizers:
                self.vosk_recognizers[ssrc] = KaldiRecognizer(self.vosk_model, SAMPLE_RATE)
            
            rec = self.vosk_recognizers[ssrc]
            
            if rec.AcceptWaveform(audio):
                result = json.loads(rec.Result())
                return result.get("text", "").strip()
            else:
                partial = json.loads(rec.PartialResult())
                text = partial.get("partial", "").strip()
                if text:
                    return f"[partial] {text}"
        except Exception as e:
            logger.error(f"Vosk processing error: {e}")
        return None
        
    def register_callback(self, ssrc: int, callback: Callable[[str, bool], Any]) -> None:
        """Register transcript callback for session"""
        self.callbacks[ssrc] = callback
        
    def unregister(self, ssrc: int) -> None:
        """Clean up session"""
        self.callbacks.pop(ssrc, None)
        self.transcripts.pop(ssrc, None)
        if hasattr(self, 'vosk_recognizers'):
            self.vosk_recognizers.pop(ssrc, None)


# ============ Global Server Instance ============

_rtp_server: Optional[RTPServer] = None
_stt_bridge: Optional[RTPSttBridge] = None


async def start_rtp_server(port: int = DEFAULT_PORT, use_riva: bool = True) -> RTPServer:
    """Start the global RTP server"""
    global _rtp_server, _stt_bridge
    
    _stt_bridge = RTPSttBridge(use_riva=use_riva)
    
    def on_audio(ssrc: int, audio: bytes):
        text = _stt_bridge.process_audio(ssrc, audio)
        if text:
            logger.info(f"[{ssrc}] {text}")
            # Deliver to callbacks
            if ssrc in _stt_bridge.callbacks:
                _stt_bridge.callbacks[ssrc](text, not text.startswith("[partial]"))
    
    _rtp_server = RTPServer(port=port, on_audio=on_audio)
    await _rtp_server.start()
    return _rtp_server


async def stop_rtp_server() -> None:
    """Stop the global RTP server"""
    global _rtp_server
    if _rtp_server:
        await _rtp_server.stop()
        _rtp_server = None


def get_rtp_server() -> Optional[RTPServer]:
    """Get the global RTP server instance"""
    return _rtp_server


def get_stt_bridge() -> Optional[RTPSttBridge]:
    """Get the global STT bridge instance"""
    return _stt_bridge
