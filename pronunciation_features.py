"""
Pronunciation Feature Extraction for ML Model Training
Extracts per-recording features that can be used to build a pronunciation scoring model.
"""

import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path

# Import our pronunciation analyzer
from pronunciation import PronunciationAnalyzer, WordPronunciation, PronunciationAnalysis


@dataclass
class RecordingFeatures:
    """Features extracted from a single recording for ML training."""
    
    # Identifiers
    recording_id: str
    audio_path: str
    duration_seconds: float
    
    # Word counts
    total_words: int
    problem_word_count: int
    problem_word_ratio: float
    
    # Confidence distributions
    mean_word_confidence: float
    std_word_confidence: float
    min_word_confidence: float
    max_word_confidence: float
    median_word_confidence: float
    
    # Thresholded counts
    pct_words_below_50: float
    pct_words_below_60: float
    pct_words_below_70: float
    pct_words_below_80: float
    pct_words_below_90: float
    
    # Pronunciation scores (our scoring)
    mean_pronunciation_score: float
    std_pronunciation_score: float
    min_pronunciation_score: float
    
    # Speaking rate
    words_per_minute: float
    mean_word_duration_ms: float
    std_word_duration_ms: float
    
    # Issue analysis
    total_issues: int
    unique_issue_types: int
    
    # For supervised learning (optional)
    human_score: Optional[float] = None
    human_label: Optional[str] = None  # "excellent", "good", "fair", "poor"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_feature_vector(self) -> List[float]:
        """Return numeric features as a vector for ML models."""
        return [
            self.total_words,
            self.problem_word_ratio,
            self.mean_word_confidence,
            self.std_word_confidence,
            self.min_word_confidence,
            self.median_word_confidence,
            self.pct_words_below_50,
            self.pct_words_below_60,
            self.pct_words_below_70,
            self.pct_words_below_80,
            self.mean_pronunciation_score,
            self.std_pronunciation_score,
            self.min_pronunciation_score,
            self.words_per_minute,
            self.mean_word_duration_ms,
            self.total_issues,
        ]
    
    @staticmethod
    def feature_names() -> List[str]:
        """Names for the feature vector columns."""
        return [
            "total_words",
            "problem_word_ratio",
            "mean_word_confidence",
            "std_word_confidence", 
            "min_word_confidence",
            "median_word_confidence",
            "pct_words_below_50",
            "pct_words_below_60",
            "pct_words_below_70",
            "pct_words_below_80",
            "mean_pronunciation_score",
            "std_pronunciation_score",
            "min_pronunciation_score",
            "words_per_minute",
            "mean_word_duration_ms",
            "total_issues",
        ]


class PronunciationFeatureExtractor:
    """Extract ML features from audio recordings."""
    
    def __init__(self, analyzer: Optional[PronunciationAnalyzer] = None):
        self.analyzer = analyzer
    
    def _ensure_analyzer(self):
        if self.analyzer is None:
            print("Loading pronunciation analyzer...")
            self.analyzer = PronunciationAnalyzer()
    
    def extract_features(
        self, 
        audio_path: str, 
        recording_id: Optional[str] = None,
        human_score: Optional[float] = None,
        human_label: Optional[str] = None
    ) -> RecordingFeatures:
        """
        Extract all features from a single recording.
        
        Args:
            audio_path: Path to audio file
            recording_id: Optional ID (defaults to filename)
            human_score: Optional human rating (1-5) for supervised learning
            human_label: Optional human label for classification
            
        Returns:
            RecordingFeatures object with all extracted features
        """
        self._ensure_analyzer()
        
        if recording_id is None:
            recording_id = Path(audio_path).stem
        
        # Run pronunciation analysis
        analysis = self.analyzer.analyze(audio_path)
        
        if not analysis.words:
            raise ValueError(f"No words detected in {audio_path}")
        
        # Extract word-level metrics
        confidences = [w.whisper_confidence for w in analysis.words]
        scores = [w.pronunciation_score for w in analysis.words]
        durations = [(w.end - w.start) * 1000 for w in analysis.words]  # ms
        
        # Calculate duration
        duration_seconds = analysis.words[-1].end if analysis.words else 0
        
        # Count issues
        all_issues = []
        for w in analysis.words:
            all_issues.extend(w.issues)
        
        return RecordingFeatures(
            recording_id=recording_id,
            audio_path=audio_path,
            duration_seconds=duration_seconds,
            
            total_words=len(analysis.words),
            problem_word_count=len(analysis.problem_words),
            problem_word_ratio=len(analysis.problem_words) / len(analysis.words) if analysis.words else 0,
            
            mean_word_confidence=float(np.mean(confidences)),
            std_word_confidence=float(np.std(confidences)),
            min_word_confidence=float(np.min(confidences)),
            max_word_confidence=float(np.max(confidences)),
            median_word_confidence=float(np.median(confidences)),
            
            pct_words_below_50=sum(1 for c in confidences if c < 0.5) / len(confidences),
            pct_words_below_60=sum(1 for c in confidences if c < 0.6) / len(confidences),
            pct_words_below_70=sum(1 for c in confidences if c < 0.7) / len(confidences),
            pct_words_below_80=sum(1 for c in confidences if c < 0.8) / len(confidences),
            pct_words_below_90=sum(1 for c in confidences if c < 0.9) / len(confidences),
            
            mean_pronunciation_score=float(np.mean(scores)),
            std_pronunciation_score=float(np.std(scores)),
            min_pronunciation_score=float(np.min(scores)),
            
            words_per_minute=(len(analysis.words) / duration_seconds * 60) if duration_seconds > 0 else 0,
            mean_word_duration_ms=float(np.mean(durations)),
            std_word_duration_ms=float(np.std(durations)),
            
            total_issues=len(all_issues),
            unique_issue_types=len(set(all_issues)),
            
            human_score=human_score,
            human_label=human_label,
        )
    
    def extract_batch(
        self, 
        audio_paths: List[str],
        human_scores: Optional[Dict[str, float]] = None
    ) -> List[RecordingFeatures]:
        """Extract features from multiple recordings."""
        results = []
        for path in audio_paths:
            recording_id = Path(path).stem
            score = human_scores.get(recording_id) if human_scores else None
            try:
                features = self.extract_features(path, recording_id, human_score=score)
                results.append(features)
            except Exception as e:
                print(f"Error processing {path}: {e}")
        return results
    
    def to_dataframe(self, features: List[RecordingFeatures]):
        """Convert features list to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame([f.to_dict() for f in features])
    
    def save_features(self, features: List[RecordingFeatures], output_path: str):
        """Save features to JSON file."""
        with open(output_path, 'w') as f:
            json.dump([f.to_dict() for f in features], f, indent=2)
        print(f"Saved {len(features)} recordings to {output_path}")


def extract_recording_features(audio_path: str) -> dict:
    """Convenience function for API integration."""
    extractor = PronunciationFeatureExtractor()
    features = extractor.extract_features(audio_path)
    return features.to_dict()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pronunciation_features.py <audio_file> [audio_file2 ...]")
        print("\nExtracts ML features for pronunciation scoring model training.")
        sys.exit(1)
    
    extractor = PronunciationFeatureExtractor()
    
    for audio_path in sys.argv[1:]:
        print(f"\n{'='*60}")
        print(f"Processing: {audio_path}")
        print('='*60)
        
        features = extractor.extract_features(audio_path)
        print(json.dumps(features.to_dict(), indent=2))
