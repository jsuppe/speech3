"""
Reading Fluency Analysis Module

Compares transcribed speech to reference text to identify:
- Word accuracy (correct, substituted, inserted, deleted)
- Hesitations and self-corrections
- Reading pace and fluency metrics
- Problem areas for improvement
"""

import re
import difflib
import statistics
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger("speechscore.reading_analysis")


@dataclass
class WordComparison:
    """Result of comparing a single word position"""
    position: int  # Position in reference text
    reference_word: Optional[str]  # Expected word (None if insertion)
    spoken_word: Optional[str]  # What was said (None if deletion)
    status: str  # 'correct', 'substitution', 'insertion', 'deletion'
    similarity: float  # 0-1, how similar the words are (for partial matches)


@dataclass
class ReadingAnalysisResult:
    """Complete reading analysis result"""
    # Overall metrics
    accuracy_percent: float  # % of words read correctly
    wpm: float  # Words per minute
    total_words: int  # Words in reference
    words_correct: int
    words_substituted: int
    words_inserted: int
    words_deleted: int
    
    # Detailed comparisons
    word_comparisons: List[Dict]  # Serializable version of WordComparison
    
    # Problem areas
    problem_words: List[Dict]  # Words that were consistently wrong
    hesitation_points: List[Dict]  # Where long pauses occurred
    
    # Fluency metrics
    fluency_score: float  # 0-100 overall fluency
    pace_consistency: float  # How consistent the reading pace was
    expression_score: float  # Prosody/intonation (placeholder)
    
    # Recommendations
    focus_areas: List[str]  # What to practice


def normalize_word(word: str) -> str:
    """Normalize a word for comparison (lowercase, remove punctuation)"""
    # Remove all punctuation and convert to lowercase
    normalized = re.sub(r'[^\w]', '', word.lower().strip())
    # Handle common contractions and variants
    contractions = {
        "dont": "don't", "doesnt": "doesn't", "didnt": "didn't",
        "cant": "can't", "wont": "won't", "wouldnt": "wouldn't",
        "couldnt": "couldn't", "shouldnt": "shouldn't",
        "isnt": "isn't", "arent": "aren't", "wasnt": "wasn't",
        "werent": "weren't", "hasnt": "hasn't", "havent": "haven't",
        "hadnt": "hadn't", "im": "i'm", "ive": "i've", "id": "i'd",
        "youre": "you're", "youve": "you've", "youd": "you'd",
        "hes": "he's", "shes": "she's", "its": "it's",
        "theyre": "they're", "theyve": "they've", "theyd": "they'd",
        "weve": "we've", "wed": "we'd", "were": "we're",
        "thats": "that's", "whats": "what's", "whos": "who's",
        "lets": "let's", "heres": "here's", "theres": "there's",
    }
    # Return normalized (just the alphanumeric version)
    return normalized


def get_word_similarity(word1: str, word2: str) -> float:
    """Calculate similarity between two words (0-1)"""
    w1 = normalize_word(word1)
    w2 = normalize_word(word2)
    
    if w1 == w2:
        return 1.0
    
    # Use sequence matcher for partial matches
    return difflib.SequenceMatcher(None, w1, w2).ratio()


def align_texts(reference: str, transcription: str) -> List[WordComparison]:
    """
    Align reference text with transcription using sequence matching.
    Returns word-by-word comparison.
    """
    # Tokenize
    ref_words = reference.split()
    trans_words = transcription.split()
    
    # Normalize for comparison
    ref_normalized = [normalize_word(w) for w in ref_words]
    trans_normalized = [normalize_word(w) for w in trans_words]
    
    # Use difflib to find optimal alignment
    matcher = difflib.SequenceMatcher(None, ref_normalized, trans_normalized)
    
    comparisons = []
    position = 0
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Words match
            for k in range(i2 - i1):
                comparisons.append(WordComparison(
                    position=position,
                    reference_word=ref_words[i1 + k],
                    spoken_word=trans_words[j1 + k],
                    status='correct',
                    similarity=1.0,
                ))
                position += 1
                
        elif tag == 'replace':
            # Words might be substituted OR very similar (Whisper transcription variations)
            ref_count = i2 - i1
            trans_count = j2 - j1
            
            # Pair up as many as possible
            for k in range(max(ref_count, trans_count)):
                ref_word = ref_words[i1 + k] if k < ref_count else None
                trans_word = trans_words[j1 + k] if k < trans_count else None
                
                if ref_word and trans_word:
                    similarity = get_word_similarity(ref_word, trans_word)
                    # High similarity (>0.85) = treat as correct (Whisper variation)
                    # Medium similarity (0.6-0.85) = substitution
                    # Low similarity (<0.6) = different word
                    if similarity >= 0.85:
                        status = 'correct'  # Close enough - likely Whisper transcription difference
                    else:
                        status = 'substitution'
                    comparisons.append(WordComparison(
                        position=position,
                        reference_word=ref_word,
                        spoken_word=trans_word,
                        status=status,
                        similarity=similarity,
                    ))
                elif ref_word:
                    # Deletion
                    comparisons.append(WordComparison(
                        position=position,
                        reference_word=ref_word,
                        spoken_word=None,
                        status='deletion',
                        similarity=0.0,
                    ))
                else:
                    # Insertion
                    comparisons.append(WordComparison(
                        position=position,
                        reference_word=None,
                        spoken_word=trans_word,
                        status='insertion',
                        similarity=0.0,
                    ))
                position += 1
                
        elif tag == 'delete':
            # Words in reference but not spoken (skipped)
            for k in range(i2 - i1):
                comparisons.append(WordComparison(
                    position=position,
                    reference_word=ref_words[i1 + k],
                    spoken_word=None,
                    status='deletion',
                    similarity=0.0,
                ))
                position += 1
                
        elif tag == 'insert':
            # Words spoken but not in reference (added)
            for k in range(j2 - j1):
                comparisons.append(WordComparison(
                    position=position,
                    reference_word=None,
                    spoken_word=trans_words[j1 + k],
                    status='insertion',
                    similarity=0.0,
                ))
                position += 1
    
    return comparisons


def find_hesitation_points(segments: List[Dict], threshold_ms: float = 800) -> List[Dict]:
    """
    Find points where the reader hesitated (long pauses mid-sentence).
    
    Args:
        segments: Whisper segments with start/end times
        threshold_ms: Pause duration to consider a hesitation (ms)
    """
    hesitations = []
    threshold_sec = threshold_ms / 1000
    
    for i in range(1, len(segments)):
        prev_seg = segments[i - 1]
        curr_seg = segments[i]
        
        gap = curr_seg.get('start', 0) - prev_seg.get('end', 0)
        
        if gap >= threshold_sec:
            hesitations.append({
                'position': i,
                'after_text': prev_seg.get('text', '').strip(),
                'before_text': curr_seg.get('text', '').strip(),
                'pause_duration': round(gap, 2),
                'timestamp': prev_seg.get('end', 0),
                'type': 'pause',
            })
    
    return hesitations


# Patterns indicating self-correction
SELF_CORRECTION_PATTERNS = [
    r'\b(I mean)\b',
    r'\b(no wait)\b',
    r'\b(sorry)\b',
    r'\b(no no)\b',
    r'\b(wait)\b',
    r'\b(actually)\b',
    r'\b(let me start over)\b',
    r'\b(start again)\b',
    r'\b(oops)\b',
]

# Filler/hesitation words in transcription
FILLER_PATTERNS = [
    r'\b(um+)\b',
    r'\b(uh+)\b',
    r'\b(er+)\b',
    r'\b(ah+)\b',
    r'\b(hmm+)\b',
    r'\b(like)\b',  # When used as filler
]


def detect_self_corrections(transcription: str) -> List[Dict]:
    """
    Detect self-corrections in the transcription.
    
    Self-corrections are when the reader realizes they made a mistake
    and verbally corrects themselves (e.g., "the cat... no, the hat").
    """
    corrections = []
    text_lower = transcription.lower()
    
    # Find explicit correction markers
    for pattern in SELF_CORRECTION_PATTERNS:
        for match in re.finditer(pattern, text_lower, re.IGNORECASE):
            # Get surrounding context
            start = max(0, match.start() - 30)
            end = min(len(transcription), match.end() + 30)
            context = transcription[start:end]
            
            corrections.append({
                'type': 'explicit_correction',
                'marker': match.group(),
                'position': match.start(),
                'context': f"...{context}...",
            })
    
    # Detect word repetitions (stuttering/restart pattern)
    # e.g., "the the dog" or "I I went"
    words = transcription.split()
    for i in range(1, len(words)):
        w1 = normalize_word(words[i-1])
        w2 = normalize_word(words[i])
        if w1 and w1 == w2 and len(w1) > 1:  # Same word repeated
            corrections.append({
                'type': 'repetition',
                'word': words[i],
                'position': i,
                'context': ' '.join(words[max(0,i-2):min(len(words),i+3)]),
            })
    
    return corrections


def detect_fillers_in_transcription(transcription: str) -> List[Dict]:
    """
    Detect filler words and hesitation sounds in the transcription.
    These indicate uncertainty or processing difficulty.
    """
    fillers = []
    text_lower = transcription.lower()
    
    for pattern in FILLER_PATTERNS:
        for match in re.finditer(pattern, text_lower, re.IGNORECASE):
            # Don't count "like" if it's used properly (approximate check)
            if match.group().lower() == 'like':
                # Check if it's the filler usage (usually followed by a noun/verb)
                before = text_lower[max(0, match.start()-15):match.start()]
                # Skip if it's "looks like", "feels like", etc.
                if any(v in before for v in ['looks', 'feels', 'sounds', 'seems', 'be', 'is', 'was']):
                    continue
            
            fillers.append({
                'type': 'filler',
                'word': match.group(),
                'position': match.start(),
            })
    
    return fillers


def calculate_expression_score(
    segments: Optional[List[Dict]] = None,
    pitch_data: Optional[Dict] = None,
    transcription: str = "",
    duration_seconds: float = 0,
) -> Tuple[float, Dict]:
    """
    Calculate expression/prosody score based on:
    - Pitch variation (monotone vs. expressive)
    - Pace variation (natural rhythm)
    - Pause patterns (appropriate punctuation pauses)
    
    Returns:
        Tuple of (score 0-100, detailed metrics)
    """
    metrics = {
        'pitch_variation': None,
        'pace_variation': None,
        'appropriate_pauses': None,
        'factors': [],
    }
    
    score = 70.0  # Default baseline
    
    # If we have pitch data from audio analysis
    if pitch_data:
        pitch_std = pitch_data.get('pitch_std', 0)
        # Good expression typically has pitch std between 20-60 Hz
        if pitch_std > 0:
            if pitch_std < 15:
                metrics['pitch_variation'] = 'monotone'
                metrics['factors'].append('Voice is quite monotone - try adding more expression')
                score -= 15
            elif pitch_std < 25:
                metrics['pitch_variation'] = 'limited'
                metrics['factors'].append('Could use more pitch variation for emphasis')
                score -= 5
            elif pitch_std <= 60:
                metrics['pitch_variation'] = 'expressive'
                score += 10
            else:
                metrics['pitch_variation'] = 'highly_variable'
                metrics['factors'].append('Pitch variation is very high - may sound unnatural')
                score -= 5
    
    # If we have segment timing for pace analysis
    if segments and len(segments) > 2:
        # Calculate pace variation between segments
        durations = []
        word_counts = []
        for seg in segments:
            duration = seg.get('end', 0) - seg.get('start', 0)
            words = len(seg.get('text', '').split())
            if duration > 0 and words > 0:
                durations.append(duration)
                word_counts.append(words)
        
        if durations:
            # Calculate WPM for each segment
            wpms = [(w / d * 60) for w, d in zip(word_counts, durations) if d > 0]
            if len(wpms) > 2:
                import statistics
                wpm_std = statistics.stdev(wpms)
                wpm_mean = statistics.mean(wpms)
                cv = (wpm_std / wpm_mean * 100) if wpm_mean > 0 else 0
                
                # Good reading has some variation but not too much
                if cv < 10:
                    metrics['pace_variation'] = 'very_consistent'
                    metrics['factors'].append('Pace is very uniform - natural reading varies more')
                elif cv < 25:
                    metrics['pace_variation'] = 'natural'
                    score += 5
                elif cv < 40:
                    metrics['pace_variation'] = 'variable'
                    score += 2
                else:
                    metrics['pace_variation'] = 'inconsistent'
                    metrics['factors'].append('Reading pace is very uneven')
                    score -= 10
    
    # Check for appropriate sentence-end pauses (punctuation)
    if segments and len(segments) > 1:
        # Look for pauses after sentence endings
        sentence_end_pauses = 0
        total_sentence_ends = 0
        
        for i in range(len(segments) - 1):
            text = segments[i].get('text', '').strip()
            if text and text[-1] in '.!?':
                total_sentence_ends += 1
                gap = segments[i+1].get('start', 0) - segments[i].get('end', 0)
                if gap >= 0.3:  # At least 300ms pause
                    sentence_end_pauses += 1
        
        if total_sentence_ends > 0:
            pause_ratio = sentence_end_pauses / total_sentence_ends
            if pause_ratio >= 0.7:
                metrics['appropriate_pauses'] = 'good'
                score += 5
            elif pause_ratio >= 0.4:
                metrics['appropriate_pauses'] = 'moderate'
            else:
                metrics['appropriate_pauses'] = 'rushed'
                metrics['factors'].append('Try pausing briefly after sentences')
                score -= 5
    
    # Cap score
    score = max(0, min(100, score))
    
    return score, metrics


def identify_problem_words(comparisons: List[WordComparison]) -> List[Dict]:
    """
    Identify words that were problematic (substituted or deleted).
    Only includes words with similarity < 0.85 (actual problems, not Whisper variations).
    """
    problems = []
    
    for comp in comparisons:
        # Only flag actual substitutions and deletions
        if comp.status == 'substitution' and comp.similarity < 0.85:
            if comp.reference_word:
                problems.append({
                    'word': comp.reference_word,
                    'status': comp.status,
                    'spoken_as': comp.spoken_word,
                    'similarity': comp.similarity,
                    'position': comp.position,
                })
        elif comp.status == 'deletion':
            if comp.reference_word:
                problems.append({
                    'word': comp.reference_word,
                    'status': comp.status,
                    'spoken_as': None,
                    'similarity': 0.0,
                    'position': comp.position,
                })
    
    return problems


def generate_focus_areas(
    accuracy: float,
    problem_words: List[Dict],
    hesitations: List[Dict],
    wpm: float,
) -> List[str]:
    """Generate recommendations based on analysis."""
    focus_areas = []
    
    # Accuracy-based recommendations
    if accuracy < 70:
        focus_areas.append("Practice with simpler texts first, then gradually increase difficulty")
    elif accuracy < 85:
        focus_areas.append("Focus on reading more slowly and carefully")
    elif accuracy < 95:
        focus_areas.append("Good accuracy! Work on difficult words individually")
    
    # Hesitation-based recommendations
    if len(hesitations) > 5:
        focus_areas.append("Practice reading ahead to reduce mid-sentence pauses")
    elif len(hesitations) > 2:
        focus_areas.append("Work on sentence flow - try to read in phrases, not word by word")
    
    # Pace recommendations
    if wpm < 100:
        focus_areas.append("Reading pace is slow - build confidence with repeated readings")
    elif wpm > 180:
        focus_areas.append("Consider slowing down slightly for better clarity")
    
    # Common problem patterns
    substitution_count = sum(1 for p in problem_words if p['status'] == 'substitution')
    deletion_count = sum(1 for p in problem_words if p['status'] == 'deletion')
    
    if substitution_count > deletion_count * 2:
        focus_areas.append("Focus on word recognition - you're substituting similar-looking words")
    elif deletion_count > substitution_count * 2:
        focus_areas.append("Focus on completeness - you're skipping words")
    
    # If doing well
    if accuracy >= 95 and len(hesitations) <= 2 and 120 <= wpm <= 160:
        focus_areas.append("Excellent reading! Try more challenging texts or focus on expression")
    
    return focus_areas[:4]  # Limit to 4 recommendations


def analyze_reading(
    reference_text: str,
    transcription: str,
    duration_seconds: float,
    segments: Optional[List[Dict]] = None,
    pitch_data: Optional[Dict] = None,
) -> Dict:
    """
    Main entry point for reading analysis.
    
    Args:
        reference_text: The original text the user was supposed to read
        transcription: What Whisper heard them say
        duration_seconds: Total recording duration
        segments: Optional Whisper segments for timing analysis
        pitch_data: Optional pitch analysis from audio processing
    
    Returns:
        Complete reading analysis as a dictionary
    """
    # Align texts
    comparisons = align_texts(reference_text, transcription)
    
    # Count metrics
    ref_word_count = len(reference_text.split())
    correct = sum(1 for c in comparisons if c.status == 'correct')
    substituted = sum(1 for c in comparisons if c.status == 'substitution')
    inserted = sum(1 for c in comparisons if c.status == 'insertion')
    deleted = sum(1 for c in comparisons if c.status == 'deletion')
    
    # Calculate accuracy (correct / total reference words)
    accuracy = (correct / ref_word_count * 100) if ref_word_count > 0 else 0
    
    # Calculate WPM from transcription
    trans_word_count = len(transcription.split())
    minutes = duration_seconds / 60
    wpm = trans_word_count / minutes if minutes > 0 else 0
    
    # Find hesitations from segment timing
    hesitations = []
    if segments:
        hesitations = find_hesitation_points(segments)
    
    # Detect self-corrections
    self_corrections = detect_self_corrections(transcription)
    
    # Detect filler words
    fillers = detect_fillers_in_transcription(transcription)
    
    # Add filler-based hesitations
    for filler in fillers:
        hesitations.append({
            'type': 'filler',
            'word': filler['word'],
            'position': filler['position'],
        })
    
    # Identify problem words
    problem_words = identify_problem_words(comparisons)
    
    # Calculate expression score
    expression_score, expression_details = calculate_expression_score(
        segments=segments,
        pitch_data=pitch_data,
        transcription=transcription,
        duration_seconds=duration_seconds,
    )
    
    # Calculate fluency score (composite)
    # Factors: accuracy (35%), pace (20%), low hesitations (20%), expression (15%), low corrections (10%)
    pace_score = 100 - min(abs(wpm - 140) * 0.5, 50)  # Ideal WPM around 140
    hesitation_penalty = min(len(hesitations) * 8, 40)  # Cap penalty at 40
    correction_penalty = min(len(self_corrections) * 5, 20)  # Cap at 20
    
    fluency_score = (
        accuracy * 0.35 +
        pace_score * 0.20 +
        max(0, 100 - hesitation_penalty) * 0.20 +
        expression_score * 0.15 +
        max(0, 100 - correction_penalty) * 0.10
    )
    fluency_score = max(0, min(100, fluency_score))
    
    # Generate focus areas (now includes self-corrections)
    focus_areas = generate_focus_areas(accuracy, problem_words, hesitations, wpm)
    
    # Add expression feedback
    if expression_details.get('factors'):
        focus_areas.extend(expression_details['factors'][:2])
    
    # Add self-correction feedback
    if len(self_corrections) > 3:
        focus_areas.append("Practice reading ahead to reduce restarts and self-corrections")
    elif len(self_corrections) > 1:
        focus_areas.append("A few restarts detected - building familiarity with the text helps")
    
    # Limit focus areas
    focus_areas = focus_areas[:5]
    
    # Serialize comparisons
    word_comparisons = [
        {
            'position': c.position,
            'reference': c.reference_word,
            'spoken': c.spoken_word,
            'status': c.status,
            'similarity': round(c.similarity, 2),
        }
        for c in comparisons
    ]
    
    return {
        # Overall metrics
        'accuracy_percent': round(accuracy, 1),
        'wpm': round(wpm, 1),
        'total_words': ref_word_count,
        'words_correct': correct,
        'words_substituted': substituted,
        'words_inserted': inserted,
        'words_deleted': deleted,
        
        # Detailed data
        'word_comparisons': word_comparisons,
        'problem_words': problem_words[:20],  # Limit to top 20
        'hesitation_points': hesitations[:15],  # Limit to 15 (now includes fillers)
        'self_corrections': self_corrections[:10],  # NEW
        'filler_count': len(fillers),  # NEW
        
        # Scores
        'fluency_score': round(fluency_score, 1),
        'pace_consistency': round(pace_score, 1),
        'expression_score': round(expression_score, 1),  # Now calculated
        'expression_details': expression_details,  # NEW - detailed breakdown
        
        # Recommendations
        'focus_areas': focus_areas,
    }


# Quick test
if __name__ == "__main__":
    ref = "The quick brown fox jumps over the lazy dog."
    trans = "The quick brown fox jumps over a lazy dog."
    
    result = analyze_reading(ref, trans, 5.0)
    print(f"Accuracy: {result['accuracy_percent']}%")
    print(f"Fluency: {result['fluency_score']}")
    print(f"Problems: {result['problem_words']}")
