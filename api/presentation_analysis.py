"""
Presentation Readiness Analysis

Analyzes speech recordings specifically for presentation preparation,
providing actionable feedback rather than raw metrics.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ReadinessLevel(Enum):
    READY = "ready"
    ALMOST = "almost"
    NEEDS_WORK = "needs_work"


@dataclass
class FillerWord:
    word: str
    timestamp: float
    
    def to_dict(self):
        return {"word": self.word, "timestamp": self.timestamp}


@dataclass
class ProblemSpot:
    start_time: float
    end_time: float
    issue_type: str  # "rushed", "slow", "low_energy", "filler_cluster", "uptalk"
    severity: str  # "minor", "moderate", "major"
    description: str
    suggestion: str
    
    def to_dict(self):
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "issue_type": self.issue_type,
            "severity": self.severity,
            "description": self.description,
            "suggestion": self.suggestion
        }


@dataclass
class TimelineSegment:
    start_time: float
    end_time: float
    status: str  # "strong", "good", "warning", "problem"
    label: str
    score: float
    
    def to_dict(self):
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status,
            "label": self.label,
            "score": self.score
        }


@dataclass
class ReadinessMetric:
    name: str
    status: str  # "pass", "warning", "fail"
    value: str
    detail: str
    icon: str
    
    def to_dict(self):
        return {
            "name": self.name,
            "status": self.status,
            "value": self.value,
            "detail": self.detail,
            "icon": self.icon
        }


@dataclass
class TopFix:
    priority: int
    title: str
    description: str
    timestamps: List[float]
    
    def to_dict(self):
        return {
            "priority": self.priority,
            "title": self.title,
            "description": self.description,
            "timestamps": self.timestamps
        }


@dataclass
class PresentationReadinessReport:
    overall_readiness: str  # "ready", "almost", "needs_work"
    readiness_score: int  # 0-100
    metrics: List[ReadinessMetric]
    filler_words: List[FillerWord]
    problem_spots: List[ProblemSpot]
    top_fixes: List[TopFix]
    timeline: List[TimelineSegment]
    duration_seconds: float
    summary: str
    
    def to_dict(self):
        return {
            "overall_readiness": self.overall_readiness,
            "readiness_score": self.readiness_score,
            "metrics": [m.to_dict() for m in self.metrics],
            "filler_words": [f.to_dict() for f in self.filler_words],
            "problem_spots": [p.to_dict() for p in self.problem_spots],
            "top_fixes": [t.to_dict() for t in self.top_fixes],
            "timeline": [t.to_dict() for t in self.timeline],
            "duration_seconds": self.duration_seconds,
            "summary": self.summary
        }


class PresentationAnalyzer:
    """Analyzes presentations for readiness and actionable improvements."""
    
    # Filler words to detect
    FILLER_WORDS = {"um", "uh", "like", "you know", "basically", "actually", 
                    "literally", "so", "right", "i mean", "kind of", "sort of"}
    
    # Ideal ranges
    IDEAL_WPM_MIN = 120
    IDEAL_WPM_MAX = 160
    IDEAL_PAUSE_RATIO = (0.15, 0.30)  # 15-30% pauses is good
    
    def analyze(
        self,
        transcript_text: str,
        segments: List[Dict],
        analysis_data: Dict,
        advanced_audio: Optional[Dict] = None
    ) -> PresentationReadinessReport:
        """
        Generate a presentation readiness report.
        
        Args:
            transcript_text: Full transcript
            segments: Word/segment timing data
            analysis_data: Standard analysis results (wpm, pitch, etc.)
            advanced_audio: Advanced audio analysis if available
        """
        
        duration = self._get_duration(segments)
        
        # Detect filler words
        filler_words = self._detect_fillers(transcript_text, segments)
        
        # Analyze pace
        wpm = analysis_data.get('wpm', 150)
        pace_metric = self._analyze_pace(wpm)
        
        # Analyze clarity
        clarity_metric = self._analyze_clarity(analysis_data)
        
        # Analyze filler words
        filler_metric = self._analyze_fillers(filler_words, duration)
        
        # Analyze energy/engagement
        energy_metric, energy_dips = self._analyze_energy(segments, advanced_audio)
        
        # Analyze confidence (uptalk, vocal fry)
        confidence_metric = self._analyze_confidence(analysis_data, advanced_audio)
        
        # Find problem spots
        problem_spots = self._find_problem_spots(
            segments, wpm, filler_words, energy_dips, advanced_audio
        )
        
        # Generate timeline
        timeline = self._generate_timeline(segments, problem_spots, duration)
        
        # Generate top fixes
        top_fixes = self._generate_top_fixes(
            filler_words, problem_spots, pace_metric, energy_metric
        )
        
        # Compile metrics
        metrics = [pace_metric, clarity_metric, filler_metric, energy_metric, confidence_metric]
        
        # Calculate overall readiness
        readiness_score, overall_readiness = self._calculate_readiness(metrics)
        
        # Generate summary
        summary = self._generate_summary(overall_readiness, metrics, top_fixes)
        
        return PresentationReadinessReport(
            overall_readiness=overall_readiness,
            readiness_score=readiness_score,
            metrics=metrics,
            filler_words=filler_words,
            problem_spots=problem_spots,
            top_fixes=top_fixes,
            timeline=timeline,
            duration_seconds=duration,
            summary=summary
        )
    
    def _get_duration(self, segments: List[Dict]) -> float:
        """Get total duration from segments."""
        if not segments:
            return 0.0
        return max(s.get('end', s.get('start', 0)) for s in segments)
    
    def _detect_fillers(self, text: str, segments: List[Dict]) -> List[FillerWord]:
        """Detect filler words with timestamps."""
        fillers = []
        text_lower = text.lower()
        
        # Build word-to-timestamp mapping from segments
        word_times = {}
        for seg in segments:
            words = seg.get('words', [])
            if words:
                for w in words:
                    word = w.get('word', '').lower().strip()
                    if word:
                        word_times[word] = w.get('start', seg.get('start', 0))
            else:
                # Segment-level
                seg_text = seg.get('text', '').lower()
                seg_start = seg.get('start', 0)
                for filler in self.FILLER_WORDS:
                    if filler in seg_text:
                        fillers.append(FillerWord(word=filler, timestamp=seg_start))
        
        # Also scan transcript for fillers
        for filler in self.FILLER_WORDS:
            count = text_lower.count(f" {filler} ") + text_lower.count(f" {filler},") + text_lower.count(f" {filler}.")
            # Estimate timestamps based on position
            if count > 0 and filler not in [f.word for f in fillers]:
                pos = 0
                duration = self._get_duration(segments)
                for _ in range(min(count, 10)):  # Cap at 10 per filler
                    idx = text_lower.find(filler, pos)
                    if idx == -1:
                        break
                    # Estimate timestamp from position
                    timestamp = (idx / len(text_lower)) * duration if duration > 0 else 0
                    fillers.append(FillerWord(word=filler, timestamp=timestamp))
                    pos = idx + 1
        
        return sorted(fillers, key=lambda f: f.timestamp)
    
    def _analyze_pace(self, wpm: float) -> ReadinessMetric:
        """Analyze speaking pace."""
        if wpm is None:
            wpm = 150  # Default
        if self.IDEAL_WPM_MIN <= wpm <= self.IDEAL_WPM_MAX:
            return ReadinessMetric(
                name="Pace",
                status="pass",
                value=f"{int(wpm)} WPM",
                detail=f"Ideal range is {self.IDEAL_WPM_MIN}-{self.IDEAL_WPM_MAX} WPM",
                icon="speed"
            )
        elif wpm < self.IDEAL_WPM_MIN:
            return ReadinessMetric(
                name="Pace",
                status="warning",
                value=f"{int(wpm)} WPM",
                detail=f"A bit slow - try {self.IDEAL_WPM_MIN}+ WPM to keep energy up",
                icon="speed"
            )
        else:
            severity = "warning" if wpm < 180 else "fail"
            return ReadinessMetric(
                name="Pace",
                status=severity,
                value=f"{int(wpm)} WPM",
                detail=f"Too fast - slow down for clarity. Aim for {self.IDEAL_WPM_MAX} WPM max",
                icon="speed"
            )
    
    def _analyze_clarity(self, analysis_data: Dict) -> ReadinessMetric:
        """Analyze speech clarity."""
        hnr = analysis_data.get('hnr_db') or 15
        jitter = analysis_data.get('jitter_pct') or 1.0
        
        # HNR > 15 is good, jitter < 2% is good
        clarity_score = min(100, (hnr / 20) * 60 + ((3 - jitter) / 3) * 40)
        
        if clarity_score >= 75:
            return ReadinessMetric(
                name="Clarity",
                status="pass",
                value=f"{int(clarity_score)}%",
                detail="Clear and easy to understand",
                icon="record_voice_over"
            )
        elif clarity_score >= 50:
            return ReadinessMetric(
                name="Clarity",
                status="warning",
                value=f"{int(clarity_score)}%",
                detail="Some words may be hard to understand",
                icon="record_voice_over"
            )
        else:
            return ReadinessMetric(
                name="Clarity",
                status="fail",
                value=f"{int(clarity_score)}%",
                detail="Work on enunciation - speak more clearly",
                icon="record_voice_over"
            )
    
    def _analyze_fillers(self, fillers: List[FillerWord], duration: float) -> ReadinessMetric:
        """Analyze filler word usage."""
        count = len(fillers)
        rate = (count / (duration / 60)) if duration > 0 else 0  # per minute
        
        if count == 0:
            return ReadinessMetric(
                name="Filler Words",
                status="pass",
                value="None detected",
                detail="Excellent - no filler words heard",
                icon="do_not_disturb"
            )
        elif rate <= 2:
            return ReadinessMetric(
                name="Filler Words",
                status="pass",
                value=f"{count} total",
                detail="Minimal filler words - good control",
                icon="do_not_disturb"
            )
        elif rate <= 5:
            top_fillers = self._get_top_fillers(fillers)
            return ReadinessMetric(
                name="Filler Words",
                status="warning",
                value=f"{count} total",
                detail=f"Try replacing {top_fillers} with pauses",
                icon="do_not_disturb"
            )
        else:
            top_fillers = self._get_top_fillers(fillers)
            return ReadinessMetric(
                name="Filler Words",
                status="fail",
                value=f"{count} total",
                detail=f"Too many fillers ({top_fillers}) - practice silent pauses",
                icon="do_not_disturb"
            )
    
    def _get_top_fillers(self, fillers: List[FillerWord]) -> str:
        """Get most common filler words."""
        counts = {}
        for f in fillers:
            counts[f.word] = counts.get(f.word, 0) + 1
        top = sorted(counts.items(), key=lambda x: -x[1])[:3]
        return ", ".join(f'"{w}"' for w, _ in top)
    
    def _analyze_energy(
        self, 
        segments: List[Dict],
        advanced_audio: Optional[Dict]
    ) -> Tuple[ReadinessMetric, List[Tuple[float, float]]]:
        """Analyze energy/engagement levels. Returns metric and list of energy dips."""
        dips = []
        
        if not segments:
            return ReadinessMetric(
                name="Energy",
                status="warning",
                value="Unknown",
                detail="Could not analyze energy levels",
                icon="bolt"
            ), dips
        
        # Check for energy variation in segments
        if advanced_audio and 'emphasis_patterns' in advanced_audio:
            emphasis = advanced_audio['emphasis_patterns']
            
            # Calculate emphasis ratio from available data
            emphasis_count = emphasis.get('emphasis_count', 0)
            emphasis_dist = emphasis.get('emphasis_distribution', '')
            emphasis_segments = emphasis.get('emphasis_segments', [])
            
            # Use distribution description if available
            if 'distributed' in emphasis_dist.lower() or 'throughout' in emphasis_dist.lower():
                status = "pass"
                detail = "Good energy variation - emphasis throughout the speech"
            elif emphasis_count >= 3 or len(emphasis_segments) >= 3:
                # Check average emphasis score
                avg_score = sum(s.get('emphasis_score', 0) for s in emphasis_segments) / max(1, len(emphasis_segments))
                if avg_score >= 0.7:
                    status = "pass"
                    detail = f"Strong emphasis moments (avg {avg_score:.0%} emphasis)"
                else:
                    status = "warning"
                    detail = "Could use more emphasis on key points"
            else:
                status = "warning"
                detail = "Could use more emphasis on key points"
            
            # Find low-energy sections
            segments_data = emphasis.get('emphasis_segments', [])
            for i, seg in enumerate(segments_data):
                if seg.get('emphasis_score', 1.0) < 0.5:
                    start = seg.get('start', i * 10)
                    # Find end of low section
                    end = start + 10  # Default 10s
                    for j in range(i + 1, len(segments_data)):
                        if segments_data[j].get('emphasis_score', 1.0) >= 0.8:
                            end = segments_data[j].get('start', end)
                            break
                    dips.append((start, end))
            
            # Calculate meaningful value
            if emphasis_segments:
                avg_score = sum(s.get('emphasis_score', 0) for s in emphasis_segments) / len(emphasis_segments)
                value = f"{avg_score:.0%} avg emphasis"
            else:
                value = f"{emphasis_count} emphasis moments"
            
            return ReadinessMetric(
                name="Energy",
                status=status,
                value=value,
                detail=detail,
                icon="bolt"
            ), dips
        
        # Fallback: use pitch variation
        pitch_std = 0
        for seg in segments:
            if 'pitch_variation' in seg:
                pitch_std = max(pitch_std, seg.get('pitch_variation', 0))
        
        if pitch_std > 50:
            status = "pass"
            detail = "Good vocal variety"
        elif pitch_std > 25:
            status = "warning"  
            detail = "Add more pitch variation"
        else:
            status = "fail"
            detail = "Too monotone - vary your pitch"
        
        return ReadinessMetric(
            name="Energy",
            status=status,
            value="Analyzed",
            detail=detail,
            icon="bolt"
        ), dips
    
    def _analyze_confidence(
        self,
        analysis_data: Dict,
        advanced_audio: Optional[Dict]
    ) -> ReadinessMetric:
        """Analyze confidence indicators (uptalk, vocal fry, etc.)."""
        
        issues = []
        
        # Check vocal fry
        vocal_fry = analysis_data.get('vocal_fry_ratio', 0)
        if vocal_fry > 0.3:
            issues.append("vocal fry")
        
        # Check uptalk from pitch contour
        if advanced_audio and 'pitch_contour' in advanced_audio:
            trends = advanced_audio['pitch_contour'].get('segment_trends', [])
            rising_endings = sum(1 for t in trends if t.get('trend') == 'rising')
            total = len(trends)
            if total > 0 and rising_endings / total > 0.4:
                issues.append("uptalk")
        
        if not issues:
            return ReadinessMetric(
                name="Confidence",
                status="pass",
                value="Strong",
                detail="Your delivery sounds confident and assured",
                icon="psychology"
            )
        elif len(issues) == 1:
            return ReadinessMetric(
                name="Confidence",
                status="warning",
                value="Good",
                detail=f"Minor issue: {issues[0]} detected",
                icon="psychology"
            )
        else:
            return ReadinessMetric(
                name="Confidence",
                status="fail",
                value="Needs work",
                detail=f"Issues: {', '.join(issues)}",
                icon="psychology"
            )
    
    def _find_problem_spots(
        self,
        segments: List[Dict],
        wpm: float,
        fillers: List[FillerWord],
        energy_dips: List[Tuple[float, float]],
        advanced_audio: Optional[Dict]
    ) -> List[ProblemSpot]:
        """Find specific problem spots in the presentation."""
        spots = []
        
        # Filler clusters (3+ fillers within 30 seconds)
        for i, filler in enumerate(fillers):
            nearby = [f for f in fillers if abs(f.timestamp - filler.timestamp) < 30]
            if len(nearby) >= 3:
                start = min(f.timestamp for f in nearby)
                end = max(f.timestamp for f in nearby) + 2
                # Avoid duplicates
                if not any(s.start_time == start for s in spots):
                    spots.append(ProblemSpot(
                        start_time=start,
                        end_time=end,
                        issue_type="filler_cluster",
                        severity="moderate",
                        description=f"{len(nearby)} filler words in quick succession",
                        suggestion="Take a breath and pause silently instead"
                    ))
        
        # Energy dips
        for start, end in energy_dips:
            if end - start > 5:  # Only flag dips > 5 seconds
                spots.append(ProblemSpot(
                    start_time=start,
                    end_time=end,
                    issue_type="low_energy",
                    severity="moderate",
                    description="Energy drops in this section",
                    suggestion="Add more emphasis and vocal variety here"
                ))
        
        # Rushed sections (from segment analysis)
        if advanced_audio and 'emphasis_patterns' in advanced_audio:
            seg_data = advanced_audio['emphasis_patterns'].get('emphasis_segments', [])
            for seg in seg_data:
                rate = seg.get('speaking_rate', 0)
                if rate > 4.5:  # Very fast
                    spots.append(ProblemSpot(
                        start_time=seg.get('start', 0),
                        end_time=seg.get('start', 0) + 5,
                        issue_type="rushed",
                        severity="minor" if rate < 5 else "moderate",
                        description="Speaking too quickly here",
                        suggestion="Slow down - this may be an important point"
                    ))
        
        return sorted(spots, key=lambda s: s.start_time)[:10]  # Top 10
    
    def _generate_timeline(
        self,
        segments: List[Dict],
        problem_spots: List[ProblemSpot],
        duration: float
    ) -> List[TimelineSegment]:
        """Generate a timeline visualization of the presentation."""
        if duration <= 0:
            return []
        
        # Create 10 equal segments
        num_segments = 10
        seg_duration = duration / num_segments
        timeline = []
        
        for i in range(num_segments):
            start = i * seg_duration
            end = (i + 1) * seg_duration
            
            # Check for problems in this segment
            problems = [p for p in problem_spots 
                       if not (p.end_time < start or p.start_time > end)]
            
            if problems:
                worst = max(problems, key=lambda p: {"minor": 1, "moderate": 2, "major": 3}.get(p.severity, 0))
                if worst.severity == "major":
                    status = "problem"
                    label = worst.issue_type.replace("_", " ").title()
                else:
                    status = "warning"
                    label = worst.issue_type.replace("_", " ").title()
                score = 50 if worst.severity == "major" else 70
            else:
                status = "strong"
                label = "Good"
                score = 90
            
            timeline.append(TimelineSegment(
                start_time=start,
                end_time=end,
                status=status,
                label=label,
                score=score
            ))
        
        return timeline
    
    def _generate_top_fixes(
        self,
        fillers: List[FillerWord],
        problem_spots: List[ProblemSpot],
        pace_metric: ReadinessMetric,
        energy_metric: ReadinessMetric
    ) -> List[TopFix]:
        """Generate top 3 actionable fixes."""
        fixes = []
        priority = 1
        
        # Filler words fix
        if len(fillers) >= 5:
            top_filler = max(set(f.word for f in fillers), 
                           key=lambda w: sum(1 for f in fillers if f.word == w))
            timestamps = [f.timestamp for f in fillers if f.word == top_filler][:5]
            fixes.append(TopFix(
                priority=priority,
                title=f'Cut "{top_filler}"',
                description=f"Replace with silent pauses. Found {len(fillers)} filler words total.",
                timestamps=timestamps
            ))
            priority += 1
        
        # Pace fix
        if pace_metric.status == "fail":
            fixes.append(TopFix(
                priority=priority,
                title="Slow down",
                description="You're speaking too fast for clarity. Take breaths between sentences.",
                timestamps=[]
            ))
            priority += 1
        
        # Energy fix
        if energy_metric.status in ["warning", "fail"]:
            energy_spots = [p for p in problem_spots if p.issue_type == "low_energy"]
            timestamps = [p.start_time for p in energy_spots[:3]]
            fixes.append(TopFix(
                priority=priority,
                title="Add energy variation",
                description="Some sections are monotone. Emphasize key points with volume and pitch changes.",
                timestamps=timestamps
            ))
            priority += 1
        
        # Rushed sections fix
        rushed = [p for p in problem_spots if p.issue_type == "rushed"]
        if rushed:
            timestamps = [p.start_time for p in rushed[:3]]
            fixes.append(TopFix(
                priority=priority,
                title="Slow down at key points",
                description="You rushed through some important moments. Pause before key statements.",
                timestamps=timestamps
            ))
            priority += 1
        
        return fixes[:3]  # Top 3 only
    
    def _calculate_readiness(self, metrics: List[ReadinessMetric]) -> Tuple[int, str]:
        """Calculate overall readiness score and level."""
        scores = {"pass": 100, "warning": 65, "fail": 30}
        
        total = sum(scores.get(m.status, 50) for m in metrics)
        avg = total / len(metrics) if metrics else 50
        
        if avg >= 85:
            return int(avg), "ready"
        elif avg >= 65:
            return int(avg), "almost"
        else:
            return int(avg), "needs_work"
    
    def _generate_summary(
        self,
        readiness: str,
        metrics: List[ReadinessMetric],
        fixes: List[TopFix]
    ) -> str:
        """Generate a human-readable summary."""
        passes = [m for m in metrics if m.status == "pass"]
        fails = [m for m in metrics if m.status == "fail"]
        
        if readiness == "ready":
            return f"You're ready to present! Strong on {', '.join(m.name.lower() for m in passes[:3])}."
        elif readiness == "almost":
            if fixes:
                return f"Almost there! Focus on: {fixes[0].title.lower()}."
            return "Almost ready - a few small tweaks needed."
        else:
            if fails:
                return f"Needs practice. Work on {', '.join(m.name.lower() for m in fails[:2])} before presenting."
            return "Keep practicing - you'll get there!"


# Singleton
_analyzer: Optional[PresentationAnalyzer] = None

def get_presentation_analyzer() -> PresentationAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = PresentationAnalyzer()
    return _analyzer
