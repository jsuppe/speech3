"""
Presentation Content Analysis

Analyzes the semantic content of a speech and compares it to the delivery tone
to identify alignment/misalignment between what is said and how it's said.
"""

import logging
import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import requests

logger = logging.getLogger(__name__)

# Ollama endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"


@dataclass
class ContentSegment:
    """A segment of content with its analysis."""
    start_time: float
    end_time: float
    text: str
    topic: str
    content_type: str  # "announcement", "achievement", "challenge", "loss", "opportunity", etc.
    expected_tone: str  # "somber", "energetic", "neutral", "serious", "optimistic", "urgent"
    detected_emotion: str
    alignment: str  # "match", "mismatch", "partial"
    feedback: str
    
    def to_dict(self):
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text": self.text,
            "topic": self.topic,
            "content_type": self.content_type,
            "expected_tone": self.expected_tone,
            "detected_emotion": self.detected_emotion,
            "alignment": self.alignment,
            "feedback": self.feedback
        }


@dataclass
class KeyPoint:
    """A key point from the presentation."""
    text: str
    importance: str  # "high", "medium", "low"
    
    def to_dict(self):
        return {"text": self.text, "importance": self.importance}


@dataclass
class PresentationIntent:
    """The inferred intent/purpose of the presentation."""
    primary_intent: str  # "motivate", "inform", "persuade", "celebrate", "console", etc.
    goal: str  # What the speaker is trying to achieve
    audience_expectation: str  # What the audience likely needs
    success_criteria: List[str]  # What would make this presentation effective
    
    def to_dict(self):
        return {
            "primary_intent": self.primary_intent,
            "goal": self.goal,
            "audience_expectation": self.audience_expectation,
            "success_criteria": self.success_criteria
        }


@dataclass
class SegmentFeedback:
    """Feedback for a specific segment based on intent."""
    start_time: float
    end_time: float
    text: str
    supports_intent: bool
    effectiveness: str  # "strong", "moderate", "weak", "counterproductive"
    feedback: str
    suggestions: List[str]
    # NEW: Contextual analysis
    meaning: str  # What this segment means in context
    purpose: str  # Why the speaker said this
    ideal_tone: str  # What tone this should have
    detected_tone: str  # What tone was actually detected
    tone_match: str  # "match", "partial", "mismatch"
    analysis: str  # Full analysis of delivery effectiveness
    
    def to_dict(self):
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text": self.text,
            "supports_intent": self.supports_intent,
            "effectiveness": self.effectiveness,
            "feedback": self.feedback,
            "suggestions": self.suggestions,
            "meaning": self.meaning,
            "purpose": self.purpose,
            "ideal_tone": self.ideal_tone,
            "detected_tone": self.detected_tone,
            "tone_match": self.tone_match,
            "analysis": self.analysis
        }


@dataclass
class ContentAnalysisResult:
    """Full content analysis result."""
    main_topic: str
    presentation_type: str  # "business_update", "pitch", "eulogy", "motivational", etc.
    intent: PresentationIntent  # NEW: Inferred intent
    key_points: List[KeyPoint]
    target_audience: str
    overall_message: str
    segment_feedback: List[SegmentFeedback]  # NEW: Per-segment content feedback
    content_segments: List[ContentSegment]
    unified_timeline: List[Dict]  # NEW: Merged timeline with all data
    alignment_score: int  # 0-100
    alignment_summary: str
    content_feedback: List[str]
    tone_recommendations: List[str]
    intent_alignment: str  # NEW: How well delivery matches intent
    
    def to_dict(self):
        return {
            "main_topic": self.main_topic,
            "presentation_type": self.presentation_type,
            "intent": self.intent.to_dict(),
            "key_points": [kp.to_dict() for kp in self.key_points],
            "target_audience": self.target_audience,
            "overall_message": self.overall_message,
            "segment_feedback": [sf.to_dict() for sf in self.segment_feedback],
            "content_segments": [cs.to_dict() for cs in self.content_segments],
            "unified_timeline": self.unified_timeline,
            "alignment_score": self.alignment_score,
            "alignment_summary": self.alignment_summary,
            "content_feedback": self.content_feedback,
            "tone_recommendations": self.tone_recommendations,
            "intent_alignment": self.intent_alignment
        }


class ContentAnalyzer:
    """Analyzes presentation content and tone alignment."""
    
    # Content type to expected tone mappings
    CONTENT_TONE_MAP = {
        "loss": ["somber", "respectful", "sad"],
        "death": ["somber", "respectful", "sad"],
        "tragedy": ["somber", "serious", "sad"],
        "failure": ["serious", "reflective", "neutral"],
        "challenge": ["serious", "determined", "neutral"],
        "problem": ["serious", "concerned", "urgent"],
        "achievement": ["energetic", "proud", "optimistic"],
        "success": ["energetic", "optimistic", "happy"],
        "growth": ["optimistic", "energetic", "positive"],
        "opportunity": ["optimistic", "energetic", "excited"],
        "vision": ["inspirational", "energetic", "optimistic"],
        "thanks": ["warm", "grateful", "sincere"],
        "welcome": ["warm", "friendly", "energetic"],
        "introduction": ["neutral", "clear", "professional"],
        "data": ["neutral", "clear", "professional"],
        "warning": ["serious", "urgent", "concerned"],
        "celebration": ["joyful", "energetic", "happy"],
    }
    
    # Emotion to tone mapping (from emotion detection)
    EMOTION_TONE_MAP = {
        "angry": "intense",
        "happy": "energetic",
        "sad": "somber",
        "neutral": "neutral",
        "calm": "measured",
        "fearful": "anxious",
        "surprised": "animated",
        "disgust": "negative",
    }
    
    def __init__(self):
        self.model = "llama3"
    
    def analyze(
        self,
        transcript: str,
        segments: List[Dict],
        emotion_data: Optional[List[Dict]] = None
    ) -> ContentAnalysisResult:
        """
        Analyze presentation content and tone alignment.
        
        Args:
            transcript: Full transcript text
            segments: Transcript segments with timing
            emotion_data: Emotion analysis per segment (from oratory analysis)
        """
        logger.info(f"ContentAnalyzer.analyze: {len(segments)} segments, {len(emotion_data or [])} emotion_data")
        
        # Step 1: Analyze overall content with LLM
        content_analysis = self._analyze_content_llm(transcript)
        
        # Step 2: Infer presentation intent
        intent = self._infer_intent(transcript, content_analysis)
        
        # Step 3: Segment-level content feedback based on intent
        segment_feedback = self._analyze_segments_for_intent(segments, intent, content_analysis, emotion_data)
        logger.info(f"Segment feedback count: {len(segment_feedback)}")
        
        # Step 4: Segment-level tone analysis
        content_segments = self._analyze_segments(segments, emotion_data, content_analysis)
        logger.info(f"Content segments count: {len(content_segments)}")
        
        # Step 5: Calculate alignment score
        alignment_score, alignment_summary = self._calculate_alignment(content_segments)
        
        # Step 6: Evaluate how well delivery matches intent
        intent_alignment = self._evaluate_intent_alignment(intent, content_segments, segment_feedback)
        
        # Step 7: Generate feedback
        content_feedback = self._generate_content_feedback(content_analysis, transcript, intent)
        tone_recommendations = self._generate_tone_recommendations(content_segments, intent)
        
        # Step 8: Build unified timeline (merge segment data)
        unified_timeline = self._build_unified_timeline(content_segments, segment_feedback)
        
        return ContentAnalysisResult(
            main_topic=content_analysis.get("main_topic", "General presentation"),
            presentation_type=content_analysis.get("type", "speech"),
            intent=intent,
            key_points=[KeyPoint(text=kp, importance="high") for kp in content_analysis.get("key_points", [])[:5]],
            target_audience=content_analysis.get("audience", "General"),
            overall_message=content_analysis.get("message", ""),
            segment_feedback=segment_feedback,
            content_segments=content_segments,
            unified_timeline=unified_timeline,
            alignment_score=alignment_score,
            alignment_summary=alignment_summary,
            content_feedback=content_feedback,
            tone_recommendations=tone_recommendations,
            intent_alignment=intent_alignment
        )
    
    def _analyze_content_llm(self, transcript: str) -> Dict:
        """Use LLM to analyze content."""
        prompt = f"""Analyze this speech transcript and respond in JSON format only.

Transcript:
{transcript[:3000]}

Respond with this exact JSON structure (no markdown, just JSON):
{{
    "main_topic": "one sentence describing the main topic",
    "type": "one of: business_update, pitch, eulogy, motivational, educational, announcement, celebration, other",
    "key_points": ["point 1", "point 2", "point 3"],
    "audience": "who this seems targeted at",
    "message": "the core message or takeaway in one sentence",
    "emotional_sections": [
        {{"text_snippet": "brief quote", "content_type": "loss/achievement/challenge/etc", "expected_tone": "somber/energetic/serious/etc"}}
    ]
}}"""

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("response", "{}")
                # Extract JSON from response
                json_match = re.search(r'\{[\s\S]*\}', text)
                if json_match:
                    return json.loads(json_match.group())
        except Exception as e:
            logger.warning(f"LLM content analysis failed: {e}")
        
        # Fallback: basic keyword analysis
        return self._fallback_content_analysis(transcript)
    
    def _infer_intent(self, transcript: str, content_analysis: Dict) -> PresentationIntent:
        """Infer the purpose/intent of the presentation."""
        
        prompt = f"""Analyze this speech and determine the speaker's PRIMARY INTENT.

Transcript excerpt:
{transcript[:2000]}

What is the speaker trying to accomplish? Respond in JSON only:
{{
    "primary_intent": "one of: motivate, inform, persuade, celebrate, console, warn, educate, rally, sell, apologize, announce",
    "goal": "one sentence describing what the speaker wants to achieve",
    "audience_expectation": "what the audience likely needs from this presentation",
    "success_criteria": ["criterion 1", "criterion 2", "criterion 3"]
}}"""

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("response", "{}")
                json_match = re.search(r'\{[\s\S]*\}', text)
                if json_match:
                    data = json.loads(json_match.group())
                    return PresentationIntent(
                        primary_intent=data.get("primary_intent", "inform"),
                        goal=data.get("goal", "Communicate information"),
                        audience_expectation=data.get("audience_expectation", "Clear information"),
                        success_criteria=data.get("success_criteria", [])[:5]
                    )
        except Exception as e:
            logger.warning(f"Intent inference failed: {e}")
        
        # Fallback based on presentation type
        ptype = content_analysis.get("type", "speech")
        intent_map = {
            "business_update": ("inform", "Update stakeholders on business status", "Clear data and next steps"),
            "pitch": ("persuade", "Convince audience to invest/buy", "Compelling value proposition"),
            "eulogy": ("console", "Honor and remember someone", "Comfort and shared memories"),
            "motivational": ("motivate", "Inspire action and confidence", "Hope and clear direction"),
            "educational": ("educate", "Teach new concepts or skills", "Clear explanations and examples"),
            "announcement": ("announce", "Share important news", "Clear information about changes"),
        }
        
        intent_data = intent_map.get(ptype, ("inform", "Share information", "Understanding"))
        return PresentationIntent(
            primary_intent=intent_data[0],
            goal=intent_data[1],
            audience_expectation=intent_data[2],
            success_criteria=["Clear message", "Appropriate tone", "Audience engagement"]
        )
    
    def _analyze_segments_for_intent(
        self,
        segments: List[Dict],
        intent: PresentationIntent,
        content_analysis: Dict,
        emotion_data: Optional[List[Dict]] = None
    ) -> List[SegmentFeedback]:
        """Analyze each segment for how well it supports the presentation intent."""
        
        feedback_list = []
        full_transcript = " ".join(s.get("text", "") for s in segments)
        
        # Group segments into larger chunks for analysis (every 3-5 segments)
        chunk_size = 3
        for i in range(0, len(segments), chunk_size):
            chunk = segments[i:i+chunk_size]
            if not chunk:
                continue
            
            combined_text = " ".join(s.get("text", "") for s in chunk)
            start_time = chunk[0].get("start", 0)
            end_time = chunk[-1].get("end", start_time + 10)
            
            if not combined_text.strip():
                continue
            
            # Get contextual analysis from LLM
            meaning, purpose, ideal_tone = self._analyze_segment_context(
                combined_text, full_transcript, intent
            )
            
            # Get detected emotion for this segment (with overlap matching)
            detected_tone = "neutral"
            if emotion_data:
                best_overlap = 0
                for ed in emotion_data:
                    ed_start = ed.get("start_time", 0)
                    ed_end = ed.get("end_time", 0)
                    overlap_start = max(start_time, ed_start)
                    overlap_end = min(end_time, ed_end)
                    overlap = max(0, overlap_end - overlap_start)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        detected_tone = ed.get("primary_emotion", "neutral")
            
            # Evaluate tone match
            tone_match, analysis = self._evaluate_tone_match(ideal_tone, detected_tone, intent)
            
            # Analyze this chunk against intent
            supports, effectiveness, fb, suggestions = self._evaluate_chunk_for_intent(
                combined_text, intent
            )
            
            feedback_list.append(SegmentFeedback(
                start_time=start_time,
                end_time=end_time,
                text=combined_text[:200] + ("..." if len(combined_text) > 200 else ""),
                supports_intent=supports,
                effectiveness=effectiveness,
                feedback=fb,
                suggestions=suggestions,
                meaning=meaning,
                purpose=purpose,
                ideal_tone=ideal_tone,
                detected_tone=detected_tone,
                tone_match=tone_match,
                analysis=analysis
            ))
        
        return feedback_list
    
    def _analyze_segment_context(
        self,
        segment_text: str,
        full_transcript: str,
        intent: PresentationIntent
    ) -> Tuple[str, str, str]:
        """Use LLM to analyze segment meaning, purpose, and ideal tone."""
        
        prompt = f"""Analyze this segment within the context of the full speech.

Full speech intent: {intent.primary_intent} - {intent.goal}

Segment to analyze:
"{segment_text[:300]}"

Context (full speech excerpt):
"{full_transcript[:800]}"

Respond in JSON only:
{{
    "meaning": "What this segment means in the context of the overall speech (1 sentence)",
    "purpose": "Why the speaker included this - what they're trying to accomplish (1 sentence)",
    "ideal_tone": "one of: energetic, somber, serious, warm, urgent, inspiring, reflective, proud, grateful, determined, compassionate"
}}"""

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3}
                },
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("response", "{}")
                json_match = re.search(r'\{[\s\S]*\}', text)
                if json_match:
                    data = json.loads(json_match.group())
                    return (
                        data.get("meaning", "Part of the speech message"),
                        data.get("purpose", "Supports overall intent"),
                        data.get("ideal_tone", "appropriate")
                    )
        except Exception as e:
            logger.warning(f"Segment context analysis failed: {e}")
        
        # Fallback
        return self._fallback_segment_analysis(segment_text, intent)
    
    def _fallback_segment_analysis(
        self,
        text: str,
        intent: PresentationIntent
    ) -> Tuple[str, str, str]:
        """Fallback segment analysis without LLM."""
        
        text_lower = text.lower()
        
        # Detect meaning based on keywords
        if any(w in text_lower for w in ["remember", "when", "back then", "years ago"]):
            meaning = "Reflecting on past events to provide context"
            purpose = "Build connection through shared history"
            tone = "reflective"
        elif any(w in text_lower for w in ["we must", "we need", "have to", "should"]):
            meaning = "Call to action or urgent message"
            purpose = "Motivate audience to take action"
            tone = "determined"
        elif any(w in text_lower for w in ["thank", "grateful", "appreciate"]):
            meaning = "Expression of gratitude"
            purpose = "Build goodwill and connection"
            tone = "warm"
        elif any(w in text_lower for w in ["proud", "achieve", "success", "accomplish"]):
            meaning = "Celebration of achievement"
            purpose = "Build confidence and pride"
            tone = "proud"
        elif any(w in text_lower for w in ["difficult", "challenge", "struggle", "hard"]):
            meaning = "Acknowledging challenges"
            purpose = "Show understanding and empathy"
            tone = "serious"
        elif any(w in text_lower for w in ["dream", "hope", "future", "vision"]):
            meaning = "Sharing aspirations and vision"
            purpose = "Inspire and motivate"
            tone = "inspiring"
        else:
            meaning = "Part of the core message"
            purpose = f"Support the {intent.primary_intent} intent"
            tone = "appropriate"
        
        return meaning, purpose, tone
    
    def _evaluate_tone_match(
        self,
        ideal_tone: str,
        detected_tone: str,
        intent: PresentationIntent
    ) -> Tuple[str, str]:
        """Evaluate how well detected tone matches ideal tone."""
        
        # Map detected emotions to tones
        emotion_to_tone = {
            "happy": ["energetic", "proud", "inspiring", "warm", "grateful"],
            "sad": ["somber", "reflective", "compassionate"],
            "angry": ["determined", "urgent", "serious"],
            "neutral": ["serious", "reflective", "appropriate"],
            "calm": ["warm", "reflective", "compassionate"],
        }
        
        compatible_tones = emotion_to_tone.get(detected_tone, ["appropriate"])
        
        if ideal_tone in compatible_tones or ideal_tone == "appropriate":
            return "match", f"Good delivery - {detected_tone} tone fits well with the {ideal_tone} content."
        elif detected_tone == "neutral":
            return "partial", f"Delivery is neutral but content calls for more {ideal_tone} energy."
        else:
            return "mismatch", f"Content calls for {ideal_tone} tone but delivery sounds {detected_tone}. Consider adjusting your energy."
    
    def _evaluate_chunk_for_intent(
        self,
        text: str,
        intent: PresentationIntent
    ) -> Tuple[bool, str, str, List[str]]:
        """Evaluate a text chunk against the presentation intent."""
        
        text_lower = text.lower()
        
        # Intent-specific evaluation
        if intent.primary_intent == "motivate":
            positive_words = ["can", "will", "together", "succeed", "achieve", "overcome", "strength", "believe"]
            negative_words = ["can't", "won't", "impossible", "never", "fail"]
            
            pos_count = sum(1 for w in positive_words if w in text_lower)
            neg_count = sum(1 for w in negative_words if w in text_lower)
            
            if pos_count >= 2 and neg_count == 0:
                return True, "strong", "Strong motivational language", []
            elif neg_count > pos_count:
                return False, "weak", "Too much negative language for motivational intent", [
                    "Balance challenges with solutions",
                    "Add words of encouragement"
                ]
            else:
                return True, "moderate", "Supports motivational intent", [
                    "Consider adding more inspiring language"
                ]
        
        elif intent.primary_intent == "inform":
            # Look for clear, factual language
            has_data = any(c.isdigit() for c in text) or any(w in text_lower for w in ["percent", "increase", "decrease", "total"])
            has_clarity = any(w in text_lower for w in ["specifically", "this means", "in other words", "for example"])
            
            if has_data and has_clarity:
                return True, "strong", "Clear informative content with data", []
            elif has_data:
                return True, "moderate", "Contains data but could be clearer", [
                    "Add context to help audience understand the numbers"
                ]
            else:
                return True, "moderate", "Informative content", [
                    "Consider adding specific examples or data"
                ]
        
        elif intent.primary_intent == "persuade" or intent.primary_intent == "sell":
            persuasive_words = ["benefit", "opportunity", "imagine", "results", "proven", "success", "value"]
            call_to_action = ["should", "must", "need to", "let's", "join", "invest", "try"]
            
            has_benefits = any(w in text_lower for w in persuasive_words)
            has_cta = any(w in text_lower for w in call_to_action)
            
            if has_benefits and has_cta:
                return True, "strong", "Persuasive with clear benefits and call to action", []
            elif has_benefits:
                return True, "moderate", "Shows benefits but needs stronger call to action", [
                    "Add a clear call to action"
                ]
            else:
                return True, "moderate", "Could be more persuasive", [
                    "Highlight specific benefits",
                    "Use more compelling language"
                ]
        
        elif intent.primary_intent == "console":
            empathy_words = ["understand", "feel", "sorry", "loss", "remember", "cherish", "love", "together"]
            harsh_words = ["should have", "failed", "mistake", "blame"]
            
            has_empathy = any(w in text_lower for w in empathy_words)
            has_harsh = any(w in text_lower for w in harsh_words)
            
            if has_empathy and not has_harsh:
                return True, "strong", "Compassionate and supportive", []
            elif has_harsh:
                return False, "counterproductive", "Language may feel judgmental", [
                    "Focus on support rather than criticism",
                    "Use gentler framing"
                ]
            else:
                return True, "moderate", "Appropriate tone", [
                    "Consider adding more empathetic language"
                ]
        
        elif intent.primary_intent == "warn":
            urgency_words = ["must", "critical", "immediately", "danger", "risk", "important", "urgent"]
            
            has_urgency = any(w in text_lower for w in urgency_words)
            
            if has_urgency:
                return True, "strong", "Conveys appropriate urgency", []
            else:
                return True, "moderate", "Could emphasize urgency more", [
                    "Make the stakes clearer",
                    "Add specific consequences"
                ]
        
        # Default evaluation
        return True, "moderate", "Supports presentation intent", []
    
    def _evaluate_intent_alignment(
        self,
        intent: PresentationIntent,
        content_segments: List[ContentSegment],
        segment_feedback: List[SegmentFeedback]
    ) -> str:
        """Evaluate how well the delivery aligns with the intent."""
        
        # Count effective segments
        strong = sum(1 for sf in segment_feedback if sf.effectiveness == "strong")
        weak = sum(1 for sf in segment_feedback if sf.effectiveness == "weak")
        counter = sum(1 for sf in segment_feedback if sf.effectiveness == "counterproductive")
        total = len(segment_feedback) if segment_feedback else 1
        
        # Count tone matches
        tone_matches = sum(1 for cs in content_segments if cs.alignment == "match")
        tone_total = len(content_segments) if content_segments else 1
        
        content_score = (strong * 100 + (total - strong - weak - counter) * 70 + weak * 40) / total
        tone_score = (tone_matches / tone_total) * 100
        
        combined = (content_score * 0.6 + tone_score * 0.4)
        
        if combined >= 80:
            return f"Excellent alignment with your {intent.primary_intent} intent. Content and delivery work together effectively."
        elif combined >= 60:
            return f"Good alignment with your {intent.primary_intent} intent. Some sections could better support your goal."
        elif combined >= 40:
            return f"Mixed alignment. Review flagged sections to better support your {intent.primary_intent} intent."
        else:
            return f"Significant misalignment with your {intent.primary_intent} intent. Consider restructuring content and adjusting delivery."
    
    def _fallback_content_analysis(self, transcript: str) -> Dict:
        """Fallback content analysis using keywords."""
        text_lower = transcript.lower()
        
        # Detect presentation type
        pres_type = "speech"
        if any(w in text_lower for w in ["revenue", "quarter", "sales", "profit", "growth"]):
            pres_type = "business_update"
        elif any(w in text_lower for w in ["invest", "opportunity", "funding", "startup"]):
            pres_type = "pitch"
        elif any(w in text_lower for w in ["passed away", "memory", "remembering", "tribute"]):
            pres_type = "eulogy"
        elif any(w in text_lower for w in ["dream", "believe", "together", "overcome"]):
            pres_type = "motivational"
        
        # Extract potential key points (sentences with key words)
        sentences = transcript.split('.')
        key_points = []
        for sent in sentences[:20]:
            if any(w in sent.lower() for w in ["important", "key", "must", "need to", "should", "goal"]):
                key_points.append(sent.strip()[:100])
        
        return {
            "main_topic": "Speech content",
            "type": pres_type,
            "key_points": key_points[:3] if key_points else ["Main point of the speech"],
            "audience": "General audience",
            "message": "",
            "emotional_sections": []
        }
    
    def _analyze_segments(
        self,
        segments: List[Dict],
        emotion_data: Optional[List[Dict]],
        content_analysis: Dict
    ) -> List[ContentSegment]:
        """Analyze each segment for content-tone alignment."""
        content_segments = []
        
        # Get emotional sections from LLM analysis
        emotional_sections = content_analysis.get("emotional_sections", [])
        
        for i, seg in enumerate(segments):
            text = seg.get("text", "")
            start = seg.get("start", 0)
            end = seg.get("end", start + 5)
            
            if not text.strip():
                continue
            
            # Detect content type for this segment
            content_type, expected_tone = self._detect_content_type(text)
            
            # Check if LLM identified this as emotional
            for es in emotional_sections:
                snippet = es.get("text_snippet", "")
                if snippet.lower() in text.lower():
                    content_type = es.get("content_type", content_type)
                    expected_tone = es.get("expected_tone", expected_tone)
                    break
            
            # Get detected emotion for this segment (with overlap matching)
            detected_emotion = "neutral"
            if emotion_data:
                best_overlap = 0
                for ed in emotion_data:
                    ed_start = ed.get("start_time", 0)
                    ed_end = ed.get("end_time", 0)
                    overlap_start = max(start, ed_start)
                    overlap_end = min(end, ed_end)
                    overlap = max(0, overlap_end - overlap_start)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        detected_emotion = ed.get("primary_emotion", "neutral")
            
            # Determine alignment
            alignment, feedback = self._check_alignment(expected_tone, detected_emotion, text)
            
            content_segments.append(ContentSegment(
                start_time=start,
                end_time=end,
                text=text,
                topic=content_type,
                content_type=content_type,
                expected_tone=expected_tone,
                detected_emotion=detected_emotion,
                alignment=alignment,
                feedback=feedback
            ))
        
        return content_segments
    
    def _detect_content_type(self, text: str) -> Tuple[str, str]:
        """Detect content type and expected tone from text."""
        text_lower = text.lower()
        
        # Check each content type
        for content_type, tones in self.CONTENT_TONE_MAP.items():
            if content_type in text_lower:
                return content_type, tones[0]
        
        # More specific patterns
        if any(w in text_lower for w in ["passed away", "died", "lost", "mourning", "rest in peace"]):
            return "loss", "somber"
        if any(w in text_lower for w in ["congratulations", "achieved", "record", "milestone"]):
            return "achievement", "energetic"
        if any(w in text_lower for w in ["unfortunately", "regret", "difficult", "bad news"]):
            return "challenge", "serious"
        if any(w in text_lower for w in ["excited", "opportunity", "new", "launch", "introducing"]):
            return "opportunity", "optimistic"
        if any(w in text_lower for w in ["thank", "grateful", "appreciate"]):
            return "thanks", "warm"
        
        return "general", "neutral"
    
    def _check_alignment(self, expected_tone: str, detected_emotion: str, text: str) -> Tuple[str, str]:
        """Check if detected emotion aligns with expected tone."""
        
        # Map detected emotion to tone category
        detected_tone = self.EMOTION_TONE_MAP.get(detected_emotion, "neutral")
        
        # Define compatible tones
        compatible = {
            "somber": ["sad", "neutral", "somber"],
            "energetic": ["happy", "energetic", "animated"],
            "serious": ["neutral", "intense", "measured"],
            "optimistic": ["happy", "energetic", "positive"],
            "neutral": ["neutral", "measured", "calm"],
            "warm": ["happy", "calm", "neutral"],
            "urgent": ["intense", "anxious", "energetic"],
        }
        
        expected_compatible = compatible.get(expected_tone, ["neutral"])
        
        if detected_emotion in expected_compatible or detected_tone == expected_tone:
            return "match", f"Good tone for {expected_tone} content"
        
        # Check for major mismatches
        bad_combos = [
            ("somber", "happy", "Too upbeat for somber content - show more empathy"),
            ("energetic", "sad", "Content is exciting but delivery sounds subdued - add energy"),
            ("serious", "happy", "Serious content needs more gravity in delivery"),
            ("urgent", "calm", "Urgent content should have more intensity"),
        ]
        
        for exp, det, feedback in bad_combos:
            if expected_tone == exp and detected_emotion == det:
                return "mismatch", feedback
        
        # Partial mismatch
        if detected_emotion == "neutral" and expected_tone != "neutral":
            return "partial", f"Consider adding more {expected_tone} tone for this content"
        
        return "partial", f"Delivery tone ({detected_emotion}) could better match content tone ({expected_tone})"
    
    def _calculate_alignment(self, segments: List[ContentSegment]) -> Tuple[int, str]:
        """Calculate overall alignment score."""
        if not segments:
            return 50, "Unable to analyze alignment"
        
        matches = sum(1 for s in segments if s.alignment == "match")
        partials = sum(1 for s in segments if s.alignment == "partial")
        mismatches = sum(1 for s in segments if s.alignment == "mismatch")
        total = len(segments)
        
        score = int((matches * 100 + partials * 60 + mismatches * 20) / total)
        
        if score >= 85:
            summary = "Excellent! Your delivery tone matches your content well."
        elif score >= 70:
            summary = "Good alignment overall with some sections that could be improved."
        elif score >= 50:
            summary = "Mixed alignment - review the flagged sections for tone adjustments."
        else:
            summary = "Significant tone mismatches detected - consider how your delivery matches your message."
        
        return score, summary
    
    def _generate_content_feedback(self, analysis: Dict, transcript: str, intent: PresentationIntent) -> List[str]:
        """Generate feedback on content itself based on intent."""
        feedback = []
        
        key_points = analysis.get("key_points", [])
        if len(key_points) < 2:
            feedback.append("Consider adding more clear key points to strengthen your message")
        elif len(key_points) > 5:
            feedback.append("You have many points - consider focusing on 3-5 key messages")
        
        # Check for clear structure
        words = transcript.lower()
        if "first" in words or "second" in words or "finally" in words:
            feedback.append("Good use of structural markers (first, second, finally)")
        else:
            feedback.append("Consider using structural markers to guide your audience")
        
        # Intent-specific feedback
        if intent.primary_intent == "persuade":
            if "because" not in words and "reason" not in words:
                feedback.append("Add reasoning to strengthen your persuasive argument")
            if not any(w in words for w in ["you", "your"]):
                feedback.append("Make it personal - use 'you' and 'your' to connect with audience")
        
        elif intent.primary_intent == "motivate":
            if not any(w in words for w in ["we", "together", "our"]):
                feedback.append("Use inclusive language ('we', 'together') to build unity")
        
        elif intent.primary_intent == "inform":
            if not any(c.isdigit() for c in transcript):
                feedback.append("Consider adding specific data or statistics to support your points")
        
        elif intent.primary_intent == "console":
            if "thank" not in words:
                feedback.append("Consider acknowledging the audience's feelings or situation")
        
        # Check for call to action based on intent
        needs_cta = intent.primary_intent in ["persuade", "sell", "motivate", "rally"]
        if needs_cta:
            if any(w in words for w in ["please", "let's", "i ask", "call to", "take action", "join"]):
                feedback.append("Strong call to action detected - good for your intent")
            else:
                feedback.append(f"Add a clear call to action to achieve your {intent.primary_intent} goal")
        
        return feedback
    
    def _generate_tone_recommendations(self, segments: List[ContentSegment], intent: PresentationIntent) -> List[str]:
        """Generate tone recommendations based on mismatches and intent."""
        recommendations = []
        
        # Intent-specific tone guidance
        intent_tone_map = {
            "motivate": "energetic and inspiring",
            "inform": "clear and measured",
            "persuade": "confident and compelling",
            "console": "warm and empathetic",
            "warn": "serious and urgent",
            "celebrate": "joyful and enthusiastic",
            "educate": "patient and clear",
            "rally": "passionate and energizing"
        }
        
        ideal_tone = intent_tone_map.get(intent.primary_intent, "appropriate")
        recommendations.append(f"For your {intent.primary_intent} intent, aim for a {ideal_tone} tone overall")
        
        mismatches = [s for s in segments if s.alignment == "mismatch"]
        partials = [s for s in segments if s.alignment == "partial"]
        
        for seg in mismatches[:3]:
            rec = f"At {seg.start_time:.0f}s: {seg.feedback}"
            recommendations.append(rec)
        
        if len(partials) > 3:
            recommendations.append(f"{len(partials)} sections could better match your {intent.primary_intent} intent")
        
        return recommendations

    def _build_unified_timeline(
        self, 
        content_segments: List[ContentSegment], 
        segment_feedback: List[SegmentFeedback]
    ) -> List[Dict]:
        """
        Build a unified timeline merging content_segments with segment_feedback.
        Uses content_segments as the base (finer granularity) and enriches with
        feedback data from overlapping segment_feedback entries.
        """
        timeline = []
        
        for cs in content_segments:
            # Find overlapping segment_feedback
            overlapping_fb = None
            best_overlap = 0
            
            for fb in segment_feedback:
                overlap_start = max(cs.start_time, fb.start_time)
                overlap_end = min(cs.end_time, fb.end_time)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    overlapping_fb = fb
            
            # Build unified entry
            entry = {
                # Timing
                "start_time": cs.start_time,
                "end_time": cs.end_time,
                "duration": cs.end_time - cs.start_time,
                
                # Text
                "text": cs.text,
                
                # Content analysis
                "content_type": cs.content_type,
                "topic": cs.topic,
                
                # Tone analysis
                "detected_tone": cs.detected_emotion,
                "expected_tone": cs.expected_tone,
                "alignment": cs.alignment,
                "tone_feedback": cs.feedback,
                
                # Intent analysis (from segment_feedback if available)
                "supports_intent": overlapping_fb.supports_intent if overlapping_fb else None,
                "effectiveness": overlapping_fb.effectiveness if overlapping_fb else None,
                "ideal_tone": overlapping_fb.ideal_tone if overlapping_fb else None,
                "tone_match": overlapping_fb.tone_match if overlapping_fb else None,
                
                # Detailed feedback
                "meaning": overlapping_fb.meaning if overlapping_fb else None,
                "purpose": overlapping_fb.purpose if overlapping_fb else None,
                "suggestions": overlapping_fb.suggestions if overlapping_fb else [],
                "analysis": overlapping_fb.analysis if overlapping_fb else None,
            }
            
            timeline.append(entry)
        
        return timeline


# Singleton
_analyzer: Optional[ContentAnalyzer] = None

def get_content_analyzer() -> ContentAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = ContentAnalyzer()
    return _analyzer
