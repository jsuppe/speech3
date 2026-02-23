"""
Real-time Pronunciation Analysis via WebSocket

Streams audio from client, provides live phoneme-level feedback
as user reads a known target text (e.g., tongue twisters).

Architecture:
- Client sends: target_text (once), then audio chunks (100ms each)
- Server sends: word-by-word pronunciation scores as detected

No Whisper needed - we use forced alignment against known text.
"""

import asyncio
import json
import logging
import struct
import io
import numpy as np
import torch
import torchaudio
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from gruut import sentences
from nltk.corpus import cmudict
import re

logger = logging.getLogger("speechscore.realtime")

router = APIRouter(prefix="/ws", tags=["realtime"])

# IPA to ARPABET mapping
IPA_TO_ARPABET = {
    'ɑ': 'AA', 'æ': 'AE', 'ʌ': 'AH', 'ɔ': 'AO', 'aʊ': 'AW', 'ə': 'AH',
    'aɪ': 'AY', 'ɛ': 'EH', 'ɝ': 'ER', 'eɪ': 'EY', 'ɪ': 'IH', 'i': 'IY',
    'oʊ': 'OW', 'ɔɪ': 'OY', 'ʊ': 'UH', 'u': 'UW', 'b': 'B', 'd': 'D',
    'ð': 'DH', 'f': 'F', 'ɡ': 'G', 'h': 'HH', 'dʒ': 'JH', 'k': 'K',
    'l': 'L', 'm': 'M', 'n': 'N', 'ŋ': 'NG', 'p': 'P', 'ɹ': 'R',
    's': 'S', 'ʃ': 'SH', 't': 'T', 'θ': 'TH', 'v': 'V', 'w': 'W',
    'j': 'Y', 'z': 'Z', 'ʒ': 'ZH', 'tʃ': 'CH'
}

# Phoneme to expected wav2vec2 characters (approximate mapping)
PHONEME_TO_CHARS = {
    'AA': 'A', 'AE': 'A', 'AH': 'A', 'AO': 'O', 'AW': 'OW', 'AY': 'I',
    'B': 'B', 'CH': 'CH', 'D': 'D', 'DH': 'TH', 'EH': 'E', 'ER': 'ER',
    'EY': 'A', 'F': 'F', 'G': 'G', 'HH': 'H', 'IH': 'I', 'IY': 'E',
    'JH': 'J', 'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 'NG': 'NG',
    'OW': 'O', 'OY': 'OY', 'P': 'P', 'R': 'R', 'S': 'S', 'SH': 'SH',
    'T': 'T', 'TH': 'TH', 'UH': 'U', 'UW': 'U', 'V': 'V', 'W': 'W',
    'Y': 'Y', 'Z': 'Z', 'ZH': 'ZH'
}


@dataclass
class WordExpectation:
    """Pre-computed expectations for a word"""
    word: str
    index: int
    phonemes: List[str]
    expected_chars: str  # What wav2vec2 should output


@dataclass 
class WordResult:
    """Result for a single word"""
    word: str
    index: int
    score: float  # 0-1
    phoneme_scores: List[Dict]  # Per-phoneme breakdown
    detected_chars: str
    status: str  # "pending", "speaking", "complete"


class RealtimePronunciationSession:
    """Manages a single real-time pronunciation session"""
    
    def __init__(self, target_text: str, model, processor, cmu_dict):
        self.target_text = target_text
        self.model = model
        self.processor = processor
        self.cmu_dict = cmu_dict
        self.device = next(model.parameters()).device
        
        # Pre-compute word expectations
        self.words = self._prepare_words(target_text)
        self.current_word_idx = 0
        
        # Audio buffer (accumulates chunks)
        self.audio_buffer = np.array([], dtype=np.float32)
        self.sample_rate = 16000
        
        # Silence detection
        self.silence_threshold = 0.01
        self.min_word_samples = int(0.15 * self.sample_rate)  # 150ms min word
        self.silence_samples = 0
        self.word_start_sample = 0
        self.is_speaking = False
        
        # Results
        self.results: List[WordResult] = []
        
    def _prepare_words(self, text: str) -> List[WordExpectation]:
        """Pre-compute expected phonemes for each word"""
        words = []
        # Split and clean text
        raw_words = re.findall(r"[a-zA-Z']+", text)
        
        for idx, word in enumerate(raw_words):
            phonemes = self._get_phonemes(word)
            expected_chars = "".join(PHONEME_TO_CHARS.get(p, p[0]) for p in phonemes)
            words.append(WordExpectation(
                word=word,
                index=idx,
                phonemes=phonemes,
                expected_chars=expected_chars.upper()
            ))
        
        return words
    
    def _get_phonemes(self, word: str) -> List[str]:
        """Get phonemes from CMUDict or gruut"""
        word_lower = word.lower().strip(".,!?;:'\"")
        
        if word_lower in self.cmu_dict:
            return [re.sub(r'\d', '', p) for p in self.cmu_dict[word_lower][0]]
        
        # Gruut fallback
        phonemes = []
        for sent in sentences(word, lang='en-us'):
            for w in sent:
                if w.phonemes:
                    for p in w.phonemes:
                        if p in IPA_TO_ARPABET:
                            phonemes.append(IPA_TO_ARPABET[p])
                        elif p not in ['|', '‖', 'ˈ', 'ˌ']:
                            phonemes.append(p.upper())
        return phonemes if phonemes else ['UNK']
    
    def _recognize_audio(self, audio: np.ndarray) -> Tuple[str, List[float]]:
        """Run wav2vec2 on audio, return chars and per-frame confidence"""
        if len(audio) < 100:
            return "", []
        
        inputs = self.processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        probs = torch.softmax(logits, dim=-1)
        predicted_ids = torch.argmax(logits, dim=-1)
        
        # Get per-frame confidence
        frame_confidence = probs.max(dim=-1).values.squeeze().cpu().numpy()
        if isinstance(frame_confidence, np.floating):
            frame_confidence = [float(frame_confidence)]
        else:
            frame_confidence = frame_confidence.tolist()
        
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription.strip().upper(), frame_confidence
    
    def _score_word(self, expected: WordExpectation, detected: str, confidence: List[float]) -> WordResult:
        """Score a word's pronunciation"""
        if not detected:
            return WordResult(
                word=expected.word,
                index=expected.index,
                score=0.0,
                phoneme_scores=[],
                detected_chars="",
                status="complete"
            )
        
        # Calculate character alignment score
        detected_clean = detected.replace(" ", "")
        expected_chars = expected.expected_chars
        
        # Use edit distance ratio
        from difflib import SequenceMatcher
        char_score = SequenceMatcher(None, expected_chars, detected_clean).ratio()
        
        # Calculate per-phoneme scores (approximate)
        phoneme_scores = []
        chars_per_phoneme = max(1, len(detected_clean) // max(1, len(expected.phonemes)))
        
        for i, phoneme in enumerate(expected.phonemes):
            # Get corresponding detected chars
            start_idx = i * chars_per_phoneme
            end_idx = start_idx + chars_per_phoneme
            detected_segment = detected_clean[start_idx:end_idx] if start_idx < len(detected_clean) else ""
            expected_char = PHONEME_TO_CHARS.get(phoneme, phoneme[0])
            
            # Score this phoneme
            if detected_segment and expected_char in detected_segment:
                p_score = 0.9
            elif detected_segment:
                p_score = 0.5  # Partial match
            else:
                p_score = 0.2
            
            phoneme_scores.append({
                "phoneme": phoneme,
                "expected_char": expected_char,
                "detected": detected_segment,
                "score": p_score
            })
        
        # Overall score
        avg_confidence = np.mean(confidence) if confidence else 0.5
        overall_score = (char_score * 0.6 + avg_confidence * 0.4)
        
        return WordResult(
            word=expected.word,
            index=expected.index,
            score=min(1.0, overall_score),
            phoneme_scores=phoneme_scores,
            detected_chars=detected_clean,
            status="complete"
        )
    
    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[WordResult]:
        """
        Process an audio chunk, return WordResult if a word was completed.
        Uses energy-based word boundary detection.
        """
        # Append to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
        
        # Calculate energy
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        is_sound = energy > self.silence_threshold
        
        if is_sound and not self.is_speaking:
            # Speech started
            self.is_speaking = True
            self.word_start_sample = max(0, len(self.audio_buffer) - len(audio_chunk))
            self.silence_samples = 0
            return None
        
        elif not is_sound and self.is_speaking:
            # Potential word boundary
            self.silence_samples += len(audio_chunk)
            
            # 200ms of silence = word boundary
            if self.silence_samples > int(0.2 * self.sample_rate):
                self.is_speaking = False
                
                # Extract word audio
                word_end = len(self.audio_buffer) - self.silence_samples
                word_audio = self.audio_buffer[self.word_start_sample:word_end]
                
                # Only process if long enough
                if len(word_audio) >= self.min_word_samples and self.current_word_idx < len(self.words):
                    # Recognize
                    detected, confidence = self._recognize_audio(word_audio)
                    
                    # Score against expected word
                    expected = self.words[self.current_word_idx]
                    result = self._score_word(expected, detected, confidence)
                    self.results.append(result)
                    self.current_word_idx += 1
                    
                    # Clear processed audio
                    self.audio_buffer = self.audio_buffer[word_end:]
                    
                    return result
        
        return None
    
    def get_current_status(self) -> Dict:
        """Get current session status"""
        return {
            "total_words": len(self.words),
            "completed_words": self.current_word_idx,
            "current_word": self.words[self.current_word_idx].word if self.current_word_idx < len(self.words) else None,
            "is_speaking": self.is_speaking,
            "results": [asdict(r) for r in self.results]
        }


# Global model cache
_wav2vec_model = None
_wav2vec_processor = None
_cmu_dict = None


def get_models():
    """Lazy-load wav2vec2 model"""
    global _wav2vec_model, _wav2vec_processor, _cmu_dict
    
    if _wav2vec_model is None:
        logger.info("Loading wav2vec2 for realtime pronunciation...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        _wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        _wav2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)
        _wav2vec_model.eval()
        
        try:
            _cmu_dict = cmudict.dict()
        except:
            import nltk
            nltk.download('cmudict', quiet=True)
            _cmu_dict = cmudict.dict()
        
        logger.info(f"Realtime pronunciation models loaded on {device}")
    
    return _wav2vec_model, _wav2vec_processor, _cmu_dict


@router.websocket("/pronunciation-stream")
async def pronunciation_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time pronunciation analysis.
    
    Protocol:
    1. Client sends JSON: {"type": "init", "target_text": "She sells sea shells"}
    2. Server responds: {"type": "ready", "words": [...]}
    3. Client sends binary audio chunks (16kHz, mono, float32)
    4. Server sends JSON word results as detected: {"type": "word_result", ...}
    5. Client sends JSON: {"type": "end"} to finish
    6. Server sends final summary: {"type": "summary", ...}
    """
    await websocket.accept()
    logger.info("Pronunciation stream connected")
    
    session: Optional[RealtimePronunciationSession] = None
    
    try:
        model, processor, cmu_dict = get_models()
        
        while True:
            message = await websocket.receive()
            
            if "text" in message:
                # JSON message
                data = json.loads(message["text"])
                msg_type = data.get("type")
                
                if msg_type == "init":
                    target_text = data.get("target_text", "")
                    if not target_text:
                        await websocket.send_json({"type": "error", "message": "No target text provided"})
                        continue
                    
                    session = RealtimePronunciationSession(target_text, model, processor, cmu_dict)
                    await websocket.send_json({
                        "type": "ready",
                        "words": [{"word": w.word, "index": w.index, "phonemes": w.phonemes} 
                                  for w in session.words]
                    })
                    logger.info(f"Session initialized with {len(session.words)} words")
                
                elif msg_type == "end":
                    if session:
                        summary = session.get_current_status()
                        summary["type"] = "summary"
                        
                        # Calculate overall score
                        if session.results:
                            summary["overall_score"] = np.mean([r.score for r in session.results])
                        else:
                            summary["overall_score"] = 0.0
                        
                        await websocket.send_json(summary)
                    break
                
                elif msg_type == "status":
                    if session:
                        status = session.get_current_status()
                        status["type"] = "status"
                        await websocket.send_json(status)
            
            elif "bytes" in message:
                # Binary audio chunk
                if session is None:
                    await websocket.send_json({"type": "error", "message": "Session not initialized"})
                    continue
                
                # Convert bytes to float32 numpy array
                audio_bytes = message["bytes"]
                audio_chunk = np.frombuffer(audio_bytes, dtype=np.float32)
                
                # Process chunk
                result = session.process_chunk(audio_chunk)
                
                if result:
                    # Word completed - send result
                    response = asdict(result)
                    response["type"] = "word_result"
                    await websocket.send_json(response)
                    logger.debug(f"Word '{result.word}' scored {result.score:.2f}")
    
    except WebSocketDisconnect:
        logger.info("Pronunciation stream disconnected")
    except Exception as e:
        logger.error(f"Pronunciation stream error: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
    finally:
        logger.info("Pronunciation stream closed")
