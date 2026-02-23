"""
Uncanny Detector - AI-generated audio detection for SpeakFit.

Uses AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks)
to detect synthetic/deepfake audio.

Based on: https://github.com/clovaai/aasist
Paper: https://arxiv.org/abs/2110.01200
"""

import os
import sys
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Add AASIST to path
AASIST_DIR = Path("/home/melchior/aasist_research")
sys.path.insert(0, str(AASIST_DIR))

# Model configuration (from AASIST.conf)
MODEL_CONFIG = {
    "architecture": "AASIST",
    "nb_samp": 64600,  # ~4 seconds at 16kHz
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0]
}

WEIGHTS_PATH = AASIST_DIR / "models" / "weights" / "AASIST.pth"


@dataclass
class UncannyResult:
    """Result of uncanny detection analysis."""
    synthetic_score: float  # 0.0 (real) to 1.0 (synthetic)
    confidence: float  # How confident the model is
    verdict: str  # "likely_real", "uncertain", "likely_synthetic"
    raw_score: float  # Raw model output
    explanation: str  # Human-readable explanation


class UncannyDetector:
    """Detector for AI-generated/synthetic audio."""
    
    def __init__(self, device: str = None):
        """Initialize the detector with AASIST model."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._loaded = False
        
    def _load_model(self):
        """Lazy-load the AASIST model."""
        if self._loaded:
            return
            
        try:
            from models.AASIST import Model
            
            self.model = Model(MODEL_CONFIG).to(self.device)
            self.model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=self.device))
            self.model.eval()
            self._loaded = True
            logger.info(f"AASIST model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load AASIST model: {e}")
            raise
    
    def _preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio for AASIST.
        
        AASIST expects:
        - 16kHz sample rate
        - Single channel
        - Fixed length (64600 samples = ~4 seconds)
        """
        # Load audio
        audio, sr = sf.read(audio_path)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Resample to 16kHz if needed
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        # Handle length - pad or truncate
        target_len = MODEL_CONFIG["nb_samp"]
        
        if len(audio) < target_len:
            # Pad with zeros
            audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
        elif len(audio) > target_len:
            # For longer audio, analyze multiple segments and average
            # For now, just take the first segment
            audio = audio[:target_len]
        
        # Convert to tensor
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0)  # Add batch dim
        return audio_tensor.to(self.device)
    
    def analyze(self, audio_path: str) -> UncannyResult:
        """Analyze audio for synthetic/AI-generated content.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            UncannyResult with synthetic likelihood and details
        """
        self._load_model()
        
        # Preprocess audio
        audio_tensor = self._preprocess_audio(audio_path)
        
        # Run inference
        with torch.no_grad():
            _, output = self.model(audio_tensor)
            
            # AASIST outputs logits for [bonafide, spoof]
            # Apply softmax to get probabilities
            probs = torch.softmax(output, dim=1)
            
            # prob[0] = bonafide (real), prob[1] = spoof (synthetic)
            real_prob = probs[0, 0].item()
            synthetic_prob = probs[0, 1].item()
            
            # Raw score (before softmax, for the synthetic class)
            raw_score = output[0, 1].item()
        
        # Determine verdict
        if synthetic_prob > 0.8:
            verdict = "likely_synthetic"
            explanation = "High probability of AI-generated audio. Strong markers of neural vocoder artifacts detected."
        elif synthetic_prob > 0.5:
            verdict = "uncertain"
            explanation = "Audio shows some synthetic characteristics but is not conclusively AI-generated."
        else:
            verdict = "likely_real"
            explanation = "Audio appears to be genuine human speech with natural acoustic properties."
        
        # Confidence based on how far from 0.5 the prediction is
        confidence = abs(synthetic_prob - 0.5) * 2  # 0-1 scale
        
        return UncannyResult(
            synthetic_score=synthetic_prob,
            confidence=confidence,
            verdict=verdict,
            raw_score=raw_score,
            explanation=explanation
        )
    
    def analyze_segments(self, audio_path: str, segment_length: float = 4.0) -> Dict[str, Any]:
        """Analyze longer audio by breaking into segments.
        
        Args:
            audio_path: Path to audio file
            segment_length: Length of each segment in seconds
            
        Returns:
            Aggregated results with per-segment details
        """
        self._load_model()
        
        # Load full audio
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Calculate segment parameters
        segment_samples = int(segment_length * sr)
        num_segments = max(1, len(audio) // segment_samples)
        
        segment_results = []
        
        for i in range(num_segments):
            start = i * segment_samples
            end = start + segment_samples
            segment = audio[start:end]
            
            # Pad if needed
            if len(segment) < MODEL_CONFIG["nb_samp"]:
                segment = np.pad(segment, (0, MODEL_CONFIG["nb_samp"] - len(segment)), mode='constant')
            
            # Run inference
            segment_tensor = torch.FloatTensor(segment[:MODEL_CONFIG["nb_samp"]]).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                _, output = self.model(segment_tensor)
                probs = torch.softmax(output, dim=1)
                synthetic_prob = probs[0, 1].item()
                
            segment_results.append({
                "segment": i + 1,
                "start_time": start / sr,
                "end_time": min(end, len(audio)) / sr,
                "synthetic_score": synthetic_prob
            })
        
        # Aggregate results
        scores = [s["synthetic_score"] for s in segment_results]
        avg_score = np.mean(scores)
        max_score = np.max(scores)
        
        # Final verdict based on average and max
        if avg_score > 0.7 or max_score > 0.9:
            verdict = "likely_synthetic"
        elif avg_score > 0.4:
            verdict = "uncertain"
        else:
            verdict = "likely_real"
        
        return {
            "synthetic_score": avg_score,
            "max_segment_score": max_score,
            "confidence": abs(avg_score - 0.5) * 2,
            "verdict": verdict,
            "num_segments": num_segments,
            "segments": segment_results,
            "explanation": f"Analyzed {num_segments} segments. Average synthetic score: {avg_score:.1%}, max: {max_score:.1%}"
        }


# Singleton instance
_detector: Optional[UncannyDetector] = None


def get_detector() -> UncannyDetector:
    """Get or create the uncanny detector instance."""
    global _detector
    if _detector is None:
        _detector = UncannyDetector()
    return _detector


def analyze_audio(audio_path: str) -> Dict[str, Any]:
    """Convenience function to analyze an audio file.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary with analysis results
    """
    detector = get_detector()
    result = detector.analyze(audio_path)
    
    return {
        "synthetic_score": result.synthetic_score,
        "confidence": result.confidence,
        "verdict": result.verdict,
        "raw_score": result.raw_score,
        "explanation": result.explanation
    }


if __name__ == "__main__":
    # Test with a sample file
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python uncanny_detector.py <audio_file>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    print(f"Analyzing: {audio_path}")
    
    detector = UncannyDetector()
    result = detector.analyze(audio_path)
    
    print(f"\n{'='*50}")
    print(f"Synthetic Score: {result.synthetic_score:.1%}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Verdict: {result.verdict}")
    print(f"Explanation: {result.explanation}")
    print(f"{'='*50}")
