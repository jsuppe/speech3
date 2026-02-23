"""
Pronunciation Analysis Module
- Uses Whisper for word-level transcription with timestamps
- Uses gruut for expected phonemes (G2P)
- Uses wav2vec2 for acoustic character recognition
- Compares expected vs detected pronunciation
"""

import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from faster_whisper import WhisperModel
from gruut import sentences
from nltk.corpus import cmudict
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import re

# IPA to ARPABET mapping (approximate)
IPA_TO_ARPABET = {
    'ɑ': 'AA', 'æ': 'AE', 'ʌ': 'AH', 'ɔ': 'AO', 'aʊ': 'AW', 'ə': 'AH',
    'aɪ': 'AY', 'ɛ': 'EH', 'ɝ': 'ER', 'eɪ': 'EY', 'ɪ': 'IH', 'i': 'IY',
    'oʊ': 'OW', 'ɔɪ': 'OY', 'ʊ': 'UH', 'u': 'UW', 'b': 'B', 'd': 'D',
    'ð': 'DH', 'f': 'F', 'ɡ': 'G', 'h': 'HH', 'dʒ': 'JH', 'k': 'K',
    'l': 'L', 'm': 'M', 'n': 'N', 'ŋ': 'NG', 'p': 'P', 'ɹ': 'R',
    's': 'S', 'ʃ': 'SH', 't': 'T', 'θ': 'TH', 'v': 'V', 'w': 'W',
    'j': 'Y', 'z': 'Z', 'ʒ': 'ZH', 'tʃ': 'CH'
}

@dataclass
class WordPronunciation:
    word: str
    start: float
    end: float
    whisper_confidence: float
    expected_phonemes: List[str]  # From G2P
    detected_chars: str  # From wav2vec2
    pronunciation_score: float  # 0-1
    issues: List[str]

@dataclass
class PronunciationAnalysis:
    words: List[WordPronunciation]
    overall_score: float
    problem_words: List[str]
    
class PronunciationAnalyzer:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Loading models on {device}...")
        
        # Whisper for transcription
        self.whisper = WhisperModel("large-v3", device=device, compute_type="float16" if device == "cuda" else "int8")
        
        # wav2vec2 for acoustic analysis
        self.wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.wav2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)
        
        # CMUDict for reference pronunciations
        try:
            self.cmudict = cmudict.dict()
        except:
            import nltk
            nltk.download('cmudict', quiet=True)
            self.cmudict = cmudict.dict()
        
        print("Models loaded!")
    
    def get_expected_phonemes(self, word: str) -> List[str]:
        """Get expected phonemes using CMUDict or gruut fallback"""
        word_lower = word.lower().strip(".,!?;:'\"")
        
        # Try CMUDict first
        if word_lower in self.cmudict:
            # Return first pronunciation, strip stress markers
            return [re.sub(r'\d', '', p) for p in self.cmudict[word_lower][0]]
        
        # Fallback to gruut
        phonemes = []
        for sent in sentences(word, lang='en-us'):
            for w in sent:
                if w.phonemes:
                    for p in w.phonemes:
                        # Convert IPA to ARPABET-like
                        if p in IPA_TO_ARPABET:
                            phonemes.append(IPA_TO_ARPABET[p])
                        elif p not in ['|', '‖', 'ˈ', 'ˌ']:
                            phonemes.append(p.upper())
        return phonemes
    
    def extract_audio_segment(self, audio_path: str, start: float, end: float) -> torch.Tensor:
        """Extract audio segment for a word"""
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample to 16kHz if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
            sr = 16000
        
        # Extract segment with small padding
        pad = 0.05  # 50ms padding
        start_sample = max(0, int((start - pad) * sr))
        end_sample = int((end + pad) * sr)
        
        return waveform[:, start_sample:end_sample].squeeze()
    
    def recognize_segment(self, audio_segment: torch.Tensor) -> Tuple[str, float]:
        """Run wav2vec2 on audio segment, return characters and confidence"""
        inputs = self.wav2vec_processor(audio_segment, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self.wav2vec_model(**inputs).logits
            
        # Get predicted ids and confidence
        probs = torch.softmax(logits, dim=-1)
        predicted_ids = torch.argmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values.mean().item()
        
        # Decode
        transcription = self.wav2vec_processor.batch_decode(predicted_ids)[0]
        
        return transcription.strip(), confidence
    
    def score_pronunciation(self, expected: List[str], detected: str, whisper_conf: float) -> Tuple[float, List[str]]:
        """Score pronunciation by comparing expected phonemes to detected characters"""
        issues = []
        
        if not expected or not detected:
            return whisper_conf, issues
        
        # Normalize detected (remove spaces, uppercase)
        detected_clean = detected.replace(" ", "").upper()
        
        # Build expected character sequence from phonemes
        # This is approximate since we're comparing phonemes to characters
        expected_chars = "".join([p[0] if p else "" for p in expected])
        
        # Calculate similarity
        if len(detected_clean) == 0:
            return 0.3, ["Word not clearly detected"]
        
        # Simple character overlap ratio
        matches = sum(1 for c in detected_clean if c in expected_chars)
        char_score = matches / max(len(detected_clean), len(expected_chars))
        
        # Combine with Whisper confidence
        score = (char_score * 0.4 + whisper_conf * 0.6)
        
        # Identify issues
        if score < 0.5:
            issues.append(f"Significant mispronunciation (expected sounds: {expected[:3]})")
        elif score < 0.7:
            issues.append(f"Minor pronunciation issue")
        
        if whisper_conf < 0.5:
            issues.append("Unclear articulation")
        
        return min(1.0, score), issues
    
    def refine_word_boundaries(self, audio_path: str, start: float, end: float, 
                                search_window: float = 0.15) -> Tuple[float, float]:
        """
        Refine word boundaries using energy-based onset/offset detection.
        
        Whisper timestamps can be imprecise, especially for short words.
        This method looks at audio energy to find the actual word boundaries.
        
        Args:
            audio_path: Path to audio file
            start: Whisper-provided start time
            end: Whisper-provided end time
            search_window: How far to search before/after (seconds)
        
        Returns:
            Refined (start, end) tuple
        """
        try:
            waveform, sr = torchaudio.load(audio_path)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.squeeze()
            
            # Resample to 16kHz if needed
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform.unsqueeze(0)).squeeze()
                sr = 16000
            
            # Define search regions
            search_start = max(0, start - search_window)
            search_end = min(len(waveform) / sr, end + search_window)
            
            # Extract search region
            start_sample = int(search_start * sr)
            end_sample = int(search_end * sr)
            segment = waveform[start_sample:end_sample].numpy()
            
            if len(segment) < 100:
                return start, end
            
            # Calculate energy using RMS in small windows
            window_size = int(0.01 * sr)  # 10ms windows
            hop_size = int(0.005 * sr)    # 5ms hop
            
            energies = []
            for i in range(0, len(segment) - window_size, hop_size):
                window = segment[i:i + window_size]
                rms = np.sqrt(np.mean(window ** 2))
                energies.append(rms)
            
            if not energies:
                return start, end
            
            energies = np.array(energies)
            
            # Normalize and find threshold
            energy_max = energies.max()
            if energy_max < 1e-6:
                return start, end
            
            energies_norm = energies / energy_max
            threshold = 0.15  # 15% of max energy
            
            # Find onset (first point above threshold)
            onset_idx = 0
            for i, e in enumerate(energies_norm):
                if e > threshold:
                    onset_idx = max(0, i - 2)  # Small lookback
                    break
            
            # Find offset (last point above threshold)
            offset_idx = len(energies_norm) - 1
            for i in range(len(energies_norm) - 1, -1, -1):
                if energies_norm[i] > threshold:
                    offset_idx = min(len(energies_norm) - 1, i + 2)  # Small lookforward
                    break
            
            # Convert back to time
            refined_start = search_start + (onset_idx * hop_size / sr)
            refined_end = search_start + (offset_idx * hop_size / sr)
            
            # Sanity checks - don't deviate too far from Whisper
            if abs(refined_start - start) > search_window:
                refined_start = start
            if abs(refined_end - end) > search_window:
                refined_end = end
            
            # Ensure minimum duration
            if refined_end - refined_start < 0.05:
                return start, end
            
            return refined_start, refined_end
            
        except Exception as e:
            print(f"Boundary refinement failed: {e}")
            return start, end
    
    def analyze(self, audio_path: str, reference_text: Optional[str] = None) -> PronunciationAnalysis:
        """Analyze pronunciation of audio file"""
        print(f"Analyzing {audio_path}...")
        
        # Step 1: Transcribe with word timestamps
        segments, info = self.whisper.transcribe(audio_path, word_timestamps=True)
        
        words = []
        for segment in segments:
            for word_info in segment.words:
                word = word_info.word.strip()
                if not word or len(word) < 2:
                    continue
                
                # Refine word boundaries using energy detection
                refined_start, refined_end = self.refine_word_boundaries(
                    audio_path, word_info.start, word_info.end
                )
                
                # Get expected phonemes
                expected = self.get_expected_phonemes(word)
                
                # Extract and analyze audio segment using refined boundaries
                try:
                    audio_seg = self.extract_audio_segment(audio_path, refined_start, refined_end)
                    detected, wav2vec_conf = self.recognize_segment(audio_seg)
                except Exception as e:
                    detected = ""
                    wav2vec_conf = 0.0
                
                # Score
                score, issues = self.score_pronunciation(expected, detected, word_info.probability)
                
                words.append(WordPronunciation(
                    word=word,
                    start=refined_start,  # Use refined timestamps
                    end=refined_end,
                    whisper_confidence=word_info.probability,
                    expected_phonemes=expected,
                    detected_chars=detected,
                    pronunciation_score=score,
                    issues=issues
                ))
        
        # Calculate overall score
        if words:
            overall_score = sum(w.pronunciation_score for w in words) / len(words)
        else:
            overall_score = 0.0
        
        # Find problem words (score < 0.6)
        problem_words = [w.word for w in words if w.pronunciation_score < 0.6]
        
        return PronunciationAnalysis(
            words=words,
            overall_score=overall_score,
            problem_words=problem_words
        )


def analyze_pronunciation(audio_path: str) -> dict:
    """Convenience function for API integration"""
    analyzer = PronunciationAnalyzer()
    result = analyzer.analyze(audio_path)
    
    return {
        "overall_score": round(result.overall_score, 3),
        "total_words": len(result.words),
        "problem_words": result.problem_words,
        "word_details": [
            {
                "word": w.word,
                "start": round(w.start, 2),
                "end": round(w.end, 2),
                "score": round(w.pronunciation_score, 3),
                "expected_phonemes": w.expected_phonemes,
                "issues": w.issues
            }
            for w in result.words
        ]
    }


if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python pronunciation.py <audio_file>")
        sys.exit(1)
    
    result = analyze_pronunciation(sys.argv[1])
    print(json.dumps(result, indent=2))
