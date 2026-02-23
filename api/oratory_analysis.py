"""
Oratory Analysis Module
Advanced speech analysis for public speaking and oratory practice.

Features:
1. Emotion Detection - per-segment emotional classification
2. Energy Arc Analysis - speech dynamics over time
3. Emphasis Detection - stressed words and phrases
4. Rhetorical Metrics - vocal variety, rhythm, authority
5. Style Classification - delivery style categorization
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import librosa
import logging

logger = logging.getLogger(__name__)

# Emotion categories
class Emotion(str, Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    DISGUSTED = "disgusted"

@dataclass
class EmotionResult:
    """Emotion detection result for a segment."""
    emotion: str
    confidence: float
    all_emotions: Dict[str, float]  # All emotion probabilities
    start_time: float
    end_time: float

@dataclass
class SegmentAnalysis:
    """Analysis for a speech segment."""
    start_time: float
    end_time: float
    
    # Emotion
    primary_emotion: str
    emotion_confidence: float
    emotion_scores: Dict[str, float]
    
    # Energy/Dynamics
    mean_energy: float
    energy_variation: float
    peak_energy: float
    
    # Pitch
    mean_pitch: float
    pitch_range: float
    pitch_variation: float
    
    # Pacing
    speaking_rate: float  # Relative to overall
    pause_ratio: float
    
    # Emphasis
    emphasized_words: List[str]
    emphasis_score: float

@dataclass
class EnergyArc:
    """Overall energy arc of the speech."""
    arc_type: str  # "building", "declining", "flat", "wave", "climactic"
    peak_position: float  # 0-1 position of peak energy
    dynamic_range: float
    energy_curve: List[float]  # Normalized energy over time
    description: str

@dataclass
class RhetoricalMetrics:
    """Rhetorical effectiveness metrics."""
    vocal_variety_score: float  # 0-100
    rhythm_score: float  # 0-100
    authority_index: float  # 0-100
    engagement_prediction: float  # 0-100
    
    # Subscores
    pitch_dynamism: float
    volume_dynamism: float
    pacing_variation: float
    pause_effectiveness: float
    
    # Descriptions
    strengths: List[str]
    improvements: List[str]

@dataclass
class StyleClassification:
    """Delivery style classification."""
    primary_style: str
    style_scores: Dict[str, float]
    similar_to: List[str]  # Famous orators with similar style
    description: str

@dataclass
class OratoryAnalysisResult:
    """Complete oratory analysis result."""
    # Per-segment analysis
    segments: List[SegmentAnalysis]
    
    # Overall metrics
    energy_arc: EnergyArc
    rhetorical_metrics: RhetoricalMetrics
    style: StyleClassification
    
    # Emotion summary
    dominant_emotion: str
    emotion_range: float  # How varied emotions were
    emotion_transitions: int  # Number of emotion changes
    
    # Overall scores
    overall_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'segments': [asdict(s) for s in self.segments],
            'energy_arc': asdict(self.energy_arc),
            'rhetorical_metrics': asdict(self.rhetorical_metrics),
            'style': asdict(self.style),
            'dominant_emotion': self.dominant_emotion,
            'emotion_range': self.emotion_range,
            'emotion_transitions': self.emotion_transitions,
            'overall_score': self.overall_score
        }


class OratoryAnalyzer:
    """Comprehensive oratory speech analyzer."""
    
    def __init__(self):
        self._emotion_model = None
        self._emotion_processor = None
        self._device = None
        
    def _load_emotion_model(self):
        """Lazy load emotion recognition model using transformers pipeline."""
        if self._emotion_model is not None:
            return
            
        try:
            from transformers import pipeline
            import torch
            
            logger.info("Loading emotion recognition model via transformers...")
            
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            device_id = 0 if torch.cuda.is_available() else -1
            
            # Use superb emotion recognition model (trained on IEMOCAP)
            # This model classifies: angry, happy, neutral, sad
            self._emotion_model = pipeline(
                "audio-classification",
                model="superb/wav2vec2-base-superb-er",
                device=device_id
            )
            
            # Map label IDs to emotion names
            self._emotion_labels = ["neutral", "happy", "sad", "angry"]
            self._emotion_label_map = {
                "neu": "neutral", "hap": "happy", "sad": "sad", "ang": "angry",
                "LABEL_0": "neutral", "LABEL_1": "happy", "LABEL_2": "sad", "LABEL_3": "angry"
            }
            
            logger.info(f"Emotion model loaded on {self._device}")
            
        except Exception as e:
            logger.warning(f"Could not load emotion model: {e}")
            self._emotion_model = None
    
    def _detect_emotion(self, audio: np.ndarray, sr: int) -> Tuple[str, float, Dict[str, float]]:
        """Detect emotion from audio segment using transformers pipeline."""
        self._load_emotion_model()
        
        if self._emotion_model is None:
            # Fallback: prosody-based emotion inference
            return self._infer_emotion_from_prosody(audio, sr)
        
        try:
            # Resample to 16kHz if needed
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
            
            # Run emotion classification pipeline
            # The pipeline accepts raw numpy arrays
            results = self._emotion_model(audio, sampling_rate=sr, top_k=4)
            
            # Build emotion scores from results
            emotion_scores = {}
            for r in results:
                label = r['label']
                score = r['score']
                # Map label to standard emotion name
                emotion_name = self._emotion_label_map.get(label, label.lower())
                emotion_scores[emotion_name] = float(score)
            
            # Add missing emotions with low scores for compatibility
            for missing in ["calm", "fearful", "surprised", "disgust"]:
                if missing not in emotion_scores:
                    emotion_scores[missing] = 0.05
            
            # Get primary emotion (highest score)
            primary_result = results[0]
            primary_label = primary_result['label']
            primary_emotion = self._emotion_label_map.get(primary_label, primary_label.lower())
            confidence = float(primary_result['score'])
            
            return primary_emotion, confidence, emotion_scores
            
        except Exception as e:
            logger.warning(f"Emotion detection failed: {e}")
            return self._infer_emotion_from_prosody(audio, sr)
    
    def _infer_emotion_from_prosody(self, audio: np.ndarray, sr: int) -> Tuple[str, float, Dict[str, float]]:
        """Infer emotion from prosodic features (fallback)."""
        # Extract prosodic features
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if not pitch_values:
            return "neutral", 0.5, {"neutral": 0.5}
        
        pitch_mean = np.mean(pitch_values)
        pitch_std = np.std(pitch_values)
        pitch_range = np.max(pitch_values) - np.min(pitch_values)
        
        # Energy features
        rms = librosa.feature.rms(y=audio)[0]
        energy_mean = np.mean(rms)
        energy_std = np.std(rms)
        
        # Speaking rate (approximate via zero crossings)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        zcr_mean = np.mean(zcr)
        
        # Simple rule-based emotion inference
        scores = {
            "neutral": 0.3,
            "happy": 0.1,
            "sad": 0.1,
            "angry": 0.1,
            "fearful": 0.1,
            "surprised": 0.1,
            "calm": 0.1,
            "disgust": 0.1
        }
        
        # High pitch + high energy + high variation = excited/happy/angry
        if pitch_mean > 200 and energy_mean > 0.1:
            if pitch_std > 50:
                scores["happy"] += 0.3
                scores["surprised"] += 0.2
            else:
                scores["angry"] += 0.3
        
        # Low pitch + low energy = sad/calm
        elif pitch_mean < 150 and energy_mean < 0.05:
            if pitch_std < 20:
                scores["sad"] += 0.3
            else:
                scores["calm"] += 0.3
        
        # High variation = emotional
        if pitch_std > 60:
            scores["happy"] += 0.1
            scores["surprised"] += 0.1
        
        # Normalize
        total = sum(scores.values())
        scores = {k: v/total for k, v in scores.items()}
        
        primary = max(scores, key=scores.get)
        return primary, scores[primary], scores
    
    def _analyze_energy_arc(self, audio: np.ndarray, sr: int, n_segments: int = 10) -> EnergyArc:
        """Analyze the energy arc of the speech."""
        # Compute RMS energy
        rms = librosa.feature.rms(y=audio)[0]
        
        # Divide into segments
        segment_len = len(rms) // n_segments
        energy_curve = []
        
        for i in range(n_segments):
            start = i * segment_len
            end = start + segment_len if i < n_segments - 1 else len(rms)
            segment_energy = np.mean(rms[start:end])
            energy_curve.append(float(segment_energy))
        
        # Normalize
        max_energy = max(energy_curve) if energy_curve else 1
        energy_curve = [e / max_energy for e in energy_curve]
        
        # Find peak position
        peak_idx = np.argmax(energy_curve)
        peak_position = peak_idx / len(energy_curve)
        
        # Calculate dynamic range
        dynamic_range = max(energy_curve) - min(energy_curve)
        
        # Classify arc type
        first_half = np.mean(energy_curve[:len(energy_curve)//2])
        second_half = np.mean(energy_curve[len(energy_curve)//2:])
        first_third = np.mean(energy_curve[:len(energy_curve)//3])
        last_third = np.mean(energy_curve[-len(energy_curve)//3:])
        
        # Check for wave pattern (multiple peaks)
        peaks = 0
        for i in range(1, len(energy_curve) - 1):
            if energy_curve[i] > energy_curve[i-1] and energy_curve[i] > energy_curve[i+1]:
                peaks += 1
        
        # Classify based on patterns - adjusted thresholds
        if peaks >= 2 and dynamic_range > 0.25:
            arc_type = "wave"
            description = "Your energy ebbs and flows dynamically, creating an engaging wave pattern that keeps listeners attentive."
        elif peak_position > 0.65 and dynamic_range > 0.25:
            arc_type = "climactic"
            description = "Excellent! Your speech builds to a powerful climax, a hallmark of great oratory."
        elif last_third > first_third * 1.15:
            arc_type = "building"
            description = "Strong energy progression - you build momentum effectively as you speak."
        elif first_third > last_third * 1.15:
            arc_type = "declining"
            description = "Your energy decreases toward the end. Try to maintain or build energy throughout."
        elif dynamic_range > 0.35:
            arc_type = "dynamic"
            description = "Good dynamic variation in your delivery - you modulate energy effectively throughout."
        elif dynamic_range > 0.2:
            arc_type = "steady"
            description = "Consistent energy throughout with some variation. Consider adding more peaks for emphasis."
        else:
            arc_type = "flat"
            description = "Your energy stays very constant. Adding dynamic variation can increase engagement."
        
        return EnergyArc(
            arc_type=arc_type,
            peak_position=peak_position,
            dynamic_range=dynamic_range,
            energy_curve=energy_curve,
            description=description
        )
    
    def _analyze_rhetorical_metrics(
        self, 
        audio: np.ndarray, 
        sr: int,
        segments: List[SegmentAnalysis]
    ) -> RhetoricalMetrics:
        """Calculate rhetorical effectiveness metrics.
        
        Calibrated so that excellent oratory (MLK, Reagan) scores 75-90.
        """
        
        # Pitch dynamism - how varied is pitch across segments
        if segments:
            pitch_means = [s.mean_pitch for s in segments if s.mean_pitch > 0]
            pitch_ranges = [s.pitch_range for s in segments]
            if pitch_means:
                # Higher base, better scaling for expressive speakers
                pitch_std = np.std(pitch_means)
                pitch_range_avg = np.mean(pitch_ranges)
                # Great orators have pitch_std > 20 and range > 50
                pitch_dynamism = min(100, 40 + pitch_std * 1.5 + pitch_range_avg * 0.3)
            else:
                pitch_dynamism = 50
        else:
            pitch_dynamism = 50
        
        # Volume dynamism - energy variation
        if segments:
            energy_means = [s.mean_energy for s in segments]
            energy_vars = [s.energy_variation for s in segments]
            if energy_means:
                # Great speakers have significant volume swings
                energy_std = np.std(energy_means)
                energy_var_avg = np.mean(energy_vars)
                volume_dynamism = min(100, 40 + energy_std * 800 + energy_var_avg * 500)
            else:
                volume_dynamism = 50
        else:
            volume_dynamism = 50
        
        # Pacing variation - speaking rate changes
        if segments:
            rates = [s.speaking_rate for s in segments if s.speaking_rate > 0]
            if rates:
                rate_std = np.std(rates)
                # Great speakers vary pace significantly
                pacing_variation = min(100, 40 + rate_std * 300)
            else:
                pacing_variation = 50
        else:
            pacing_variation = 50
        
        # Pause effectiveness (based on pause distribution)
        if segments:
            pause_ratios = [s.pause_ratio for s in segments]
            pause_mean = np.mean(pause_ratios)
            pause_std = np.std(pause_ratios)
            # Good pauses: present (mean > 0.05), varied (std > 0.05), not excessive
            if pause_mean > 0.02:
                pause_effectiveness = min(100, 50 + pause_std * 400 + pause_mean * 100)
            else:
                pause_effectiveness = 40  # No pauses = lower score
        else:
            pause_effectiveness = 50
        
        # Emotion variation bonus - great orators show multiple emotions
        if segments:
            emotions = set(s.primary_emotion for s in segments)
            emotion_bonus = len(emotions) * 8  # Up to 32 for 4 emotions
        else:
            emotion_bonus = 0
        
        # Calculate composite scores with better weighting
        vocal_variety = min(100, (pitch_dynamism * 0.35 + volume_dynamism * 0.35 + pacing_variation * 0.3) + emotion_bonus * 0.3)
        
        # Rhythm score - regularity with intentional variation
        rhythm_score = min(100, 45 + pacing_variation * 0.25 + pause_effectiveness * 0.25 + pitch_dynamism * 0.1)
        
        # Authority index - confident, controlled delivery
        # High volume + strategic pauses + controlled pitch = authority
        authority_index = min(100, 40 + (
            volume_dynamism * 0.35 + 
            pause_effectiveness * 0.35 +
            min(pitch_dynamism, 70) * 0.15 +  # Some pitch variation is good, too much sounds uncertain
            15  # Base confidence
        ))
        
        # Engagement prediction - would audience stay attentive?
        engagement_prediction = min(100, (
            vocal_variety * 0.30 + 
            rhythm_score * 0.25 + 
            authority_index * 0.25 +
            emotion_bonus * 0.5 +  # Emotional range is highly engaging
            10  # Base engagement
        ))
        
        # Generate feedback
        strengths = []
        improvements = []
        
        if vocal_variety > 70:
            strengths.append("Excellent vocal variety keeps listeners engaged")
        elif vocal_variety < 40:
            improvements.append("Add more variation in pitch and volume")
        
        if rhythm_score > 70:
            strengths.append("Good rhythmic delivery with effective pacing")
        elif rhythm_score < 40:
            improvements.append("Work on your pacing and use strategic pauses")
        
        if authority_index > 70:
            strengths.append("Your delivery conveys confidence and authority")
        elif authority_index < 40:
            improvements.append("Project more confidence through stronger delivery")
        
        if pause_effectiveness > 70:
            strengths.append("Effective use of pauses for emphasis")
        elif pause_effectiveness < 40:
            improvements.append("Use more strategic pauses to let points land")
        
        return RhetoricalMetrics(
            vocal_variety_score=round(vocal_variety, 1),
            rhythm_score=round(rhythm_score, 1),
            authority_index=round(authority_index, 1),
            engagement_prediction=round(engagement_prediction, 1),
            pitch_dynamism=round(pitch_dynamism, 1),
            volume_dynamism=round(volume_dynamism, 1),
            pacing_variation=round(pacing_variation, 1),
            pause_effectiveness=round(pause_effectiveness, 1),
            strengths=strengths,
            improvements=improvements
        )
    
    def _classify_style(
        self, 
        segments: List[SegmentAnalysis],
        rhetorical: RhetoricalMetrics,
        dominant_emotion: str
    ) -> StyleClassification:
        """Classify the delivery style."""
        
        styles = {
            "conversational": 0.0,
            "formal": 0.0,
            "passionate": 0.0,
            "measured": 0.0,
            "authoritative": 0.0,
            "intimate": 0.0,
            "energetic": 0.0,
            "contemplative": 0.0
        }
        
        # Score based on metrics
        if rhetorical.vocal_variety_score > 70:
            styles["passionate"] += 0.3
            styles["energetic"] += 0.2
        
        if rhetorical.authority_index > 70:
            styles["authoritative"] += 0.3
            styles["formal"] += 0.2
        
        if rhetorical.pacing_variation < 30:
            styles["measured"] += 0.3
            styles["contemplative"] += 0.2
        
        if rhetorical.pause_effectiveness > 70:
            styles["measured"] += 0.2
            styles["authoritative"] += 0.1
        
        # Score based on emotion
        if dominant_emotion in ["happy", "surprised"]:
            styles["energetic"] += 0.2
            styles["passionate"] += 0.2
        elif dominant_emotion in ["calm", "neutral"]:
            styles["measured"] += 0.2
            styles["conversational"] += 0.2
        elif dominant_emotion in ["sad", "fearful"]:
            styles["intimate"] += 0.2
            styles["contemplative"] += 0.2
        elif dominant_emotion == "angry":
            styles["passionate"] += 0.3
            styles["authoritative"] += 0.2
        
        # Normalize
        total = sum(styles.values()) or 1
        styles = {k: v/total for k, v in styles.items()}
        
        # Get primary style
        primary_style = max(styles, key=styles.get)
        
        # Compare to famous orators
        similar_to = []
        if styles["passionate"] > 0.25:
            similar_to.append("Martin Luther King Jr.")
        if styles["authoritative"] > 0.25:
            similar_to.append("Winston Churchill")
        if styles["measured"] > 0.25:
            similar_to.append("Barack Obama")
        if styles["conversational"] > 0.25:
            similar_to.append("Steve Jobs")
        if styles["energetic"] > 0.25:
            similar_to.append("Tony Robbins")
        if styles["contemplative"] > 0.25:
            similar_to.append("Carl Sagan")
        
        if not similar_to:
            similar_to = ["Developing your unique style"]
        
        # Generate description
        descriptions = {
            "conversational": "Your delivery feels natural and approachable, like talking to a friend.",
            "formal": "You have a polished, professional delivery style.",
            "passionate": "Your delivery is emotionally compelling and energetic.",
            "measured": "You speak with deliberate, thoughtful pacing.",
            "authoritative": "Your delivery commands attention and respect.",
            "intimate": "You create a sense of personal connection with listeners.",
            "energetic": "Your high-energy delivery keeps audiences alert and engaged.",
            "contemplative": "Your reflective style invites listeners to think deeply."
        }
        
        return StyleClassification(
            primary_style=primary_style,
            style_scores={k: round(v, 3) for k, v in styles.items()},
            similar_to=similar_to[:3],
            description=descriptions.get(primary_style, "Developing your unique speaking style.")
        )
    
    def analyze(
        self, 
        audio_path: str,
        segment_duration: float = 0.5,
        word_timestamps: Optional[List[Dict]] = None
    ) -> OratoryAnalysisResult:
        """
        Perform complete oratory analysis on audio file.
        
        Args:
            audio_path: Path to audio file
            segment_duration: Duration of each analysis segment in seconds
            word_timestamps: Optional word-level timestamps from transcription
            
        Returns:
            OratoryAnalysisResult with all analysis data
        """
        logger.info(f"Starting oratory analysis for {audio_path}")
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        duration = len(audio) / sr
        
        # Analyze segments
        segments = []
        n_segments = max(1, int(duration / segment_duration))
        segment_samples = len(audio) // n_segments
        
        emotion_counts = {}
        
        for i in range(n_segments):
            start_sample = i * segment_samples
            end_sample = start_sample + segment_samples if i < n_segments - 1 else len(audio)
            segment_audio = audio[start_sample:end_sample]
            
            start_time = start_sample / sr
            end_time = end_sample / sr
            
            # Emotion detection
            emotion, confidence, emotion_scores = self._detect_emotion(segment_audio, sr)
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            # Energy analysis
            rms = librosa.feature.rms(y=segment_audio)[0]
            mean_energy = float(np.mean(rms))
            energy_var = float(np.std(rms))
            peak_energy = float(np.max(rms))
            
            # Pitch analysis
            pitches, magnitudes = librosa.piptrack(y=segment_audio, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                idx = magnitudes[:, t].argmax()
                pitch = pitches[idx, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                mean_pitch = float(np.mean(pitch_values))
                pitch_range = float(np.max(pitch_values) - np.min(pitch_values))
                pitch_var = float(np.std(pitch_values))
            else:
                mean_pitch = 0
                pitch_range = 0
                pitch_var = 0
            
            # Pacing (relative speaking rate via zero crossings)
            zcr = librosa.feature.zero_crossing_rate(segment_audio)[0]
            speaking_rate = float(np.mean(zcr))
            
            # Pause ratio (silence detection)
            silence_threshold = 0.01
            silence_frames = np.sum(rms < silence_threshold)
            pause_ratio = float(silence_frames / len(rms))
            
            # Emphasized words (placeholder - would need word alignment)
            emphasized_words = []
            emphasis_score = min(100, energy_var * 1000)
            
            segments.append(SegmentAnalysis(
                start_time=start_time,
                end_time=end_time,
                primary_emotion=emotion,
                emotion_confidence=confidence,
                emotion_scores=emotion_scores,
                mean_energy=mean_energy,
                energy_variation=energy_var,
                peak_energy=peak_energy,
                mean_pitch=mean_pitch,
                pitch_range=pitch_range,
                pitch_variation=pitch_var,
                speaking_rate=speaking_rate,
                pause_ratio=pause_ratio,
                emphasized_words=emphasized_words,
                emphasis_score=emphasis_score
            ))
        
        # Overall emotion analysis
        dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral"
        unique_emotions = len(set(s.primary_emotion for s in segments))
        emotion_range = unique_emotions / len(self._emotion_labels) if hasattr(self, '_emotion_labels') else unique_emotions / 8
        
        # Count emotion transitions
        transitions = 0
        for i in range(1, len(segments)):
            if segments[i].primary_emotion != segments[i-1].primary_emotion:
                transitions += 1
        
        # Analyze energy arc
        energy_arc = self._analyze_energy_arc(audio, sr)
        
        # Calculate rhetorical metrics
        rhetorical_metrics = self._analyze_rhetorical_metrics(audio, sr, segments)
        
        # Classify style
        style = self._classify_style(segments, rhetorical_metrics, dominant_emotion)
        
        # Calculate overall score
        overall_score = (
            rhetorical_metrics.vocal_variety_score * 0.25 +
            rhetorical_metrics.rhythm_score * 0.2 +
            rhetorical_metrics.authority_index * 0.2 +
            rhetorical_metrics.engagement_prediction * 0.2 +
            (energy_arc.dynamic_range * 100) * 0.15
        )
        
        logger.info(f"Oratory analysis complete. Overall score: {overall_score:.1f}")
        
        return OratoryAnalysisResult(
            segments=segments,
            energy_arc=energy_arc,
            rhetorical_metrics=rhetorical_metrics,
            style=style,
            dominant_emotion=dominant_emotion,
            emotion_range=emotion_range,
            emotion_transitions=transitions,
            overall_score=round(overall_score, 1)
        )


# Singleton instance
_analyzer: Optional[OratoryAnalyzer] = None

def get_oratory_analyzer() -> OratoryAnalyzer:
    """Get or create the oratory analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = OratoryAnalyzer()
    return _analyzer
