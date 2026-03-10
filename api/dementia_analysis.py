"""
Dementia Conversation Analysis Module

Analyzes conversations between two participants for cognitive markers:
1. Preposition usage rate (per 10 words) - lower rates correlate with cognitive decline
2. Repetition detection - repeated concepts/questions indicate memory issues

References:
- Preposition deficit in Alzheimer's: https://pubmed.ncbi.nlm.nih.gov/8931575/
- Repetitive speech patterns: https://pubmed.ncbi.nlm.nih.gov/29493412/
"""

import re
import logging
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import spacy
from difflib import SequenceMatcher

logger = logging.getLogger("speechscore.dementia")

# Sentence transformer for semantic similarity (lazy loaded)
_semantic_model = None

def get_semantic_model():
    """Lazy load sentence transformer model."""
    global _semantic_model
    if _semantic_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded semantic similarity model: all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning(f"Failed to load semantic model: {e}")
            _semantic_model = False  # Mark as failed, don't retry
    return _semantic_model if _semantic_model else None


def semantic_similarity(text1: str, text2: str) -> float:
    """
    Calculate semantic similarity between two texts using sentence embeddings.
    Falls back to string similarity if model unavailable.
    
    Returns: similarity score 0.0-1.0
    """
    model = get_semantic_model()
    if model is None:
        # Fallback to string similarity
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    try:
        embeddings = model.encode([text1, text2], convert_to_numpy=True)
        # Cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)
    except Exception as e:
        logger.warning(f"Semantic similarity failed: {e}")
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def batch_semantic_similarity(texts: List[str]) -> np.ndarray:
    """
    Compute pairwise semantic similarity matrix for a list of texts.
    Much faster than calling semantic_similarity() in nested loops.
    
    Returns: NxN similarity matrix
    """
    model = get_semantic_model()
    if model is None or len(texts) == 0:
        return np.zeros((len(texts), len(texts)))
    
    try:
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = embeddings / norms
        # Compute cosine similarity matrix
        similarity_matrix = np.dot(normalized, normalized.T)
        return similarity_matrix
    except Exception as e:
        logger.warning(f"Batch semantic similarity failed: {e}")
        return np.zeros((len(texts), len(texts)))


# Load spaCy model for POS tagging
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Common English prepositions
PREPOSITIONS = {
    'about', 'above', 'across', 'after', 'against', 'along', 'among', 'around',
    'at', 'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond',
    'by', 'down', 'during', 'except', 'for', 'from', 'in', 'inside', 'into',
    'like', 'near', 'of', 'off', 'on', 'onto', 'out', 'outside', 'over',
    'past', 'since', 'through', 'throughout', 'till', 'to', 'toward', 'towards',
    'under', 'underneath', 'until', 'up', 'upon', 'with', 'within', 'without'
}


def count_prepositions(text: str) -> Tuple[int, int, List[str]]:
    """
    Count prepositions in text.
    Returns: (preposition_count, word_count, list_of_prepositions_found)
    """
    doc = nlp(text.lower())
    words = [token.text for token in doc if token.is_alpha]
    word_count = len(words)
    
    # Use both POS tagging and word list for accuracy
    prepositions_found = []
    for token in doc:
        if token.pos_ == "ADP" or token.text.lower() in PREPOSITIONS:
            if token.is_alpha:
                prepositions_found.append(token.text.lower())
    
    return len(prepositions_found), word_count, prepositions_found


def calculate_preposition_rate(text: str) -> Dict[str, Any]:
    """Calculate prepositions per 10 words."""
    prep_count, word_count, preps = count_prepositions(text)
    
    if word_count == 0:
        return {
            "preposition_count": 0,
            "word_count": 0,
            "rate_per_10_words": 0.0,
            "prepositions_used": []
        }
    
    rate = (prep_count / word_count) * 10
    
    return {
        "preposition_count": prep_count,
        "word_count": word_count,
        "rate_per_10_words": round(rate, 2),
        "prepositions_used": list(set(preps)),
        "preposition_frequency": dict(Counter(preps))
    }


def extract_concepts(text: str) -> List[str]:
    """Extract key concepts/phrases from text for repetition detection."""
    doc = nlp(text)
    concepts = []
    
    # Extract noun phrases
    for chunk in doc.noun_chunks:
        concepts.append(chunk.text.lower().strip())
    
    # Extract verb phrases (VERB + dependencies) - catches "went to the store"
    for token in doc:
        if token.pos_ == "VERB":
            # Get the verb and its dependents
            phrase_tokens = [token] + list(token.subtree)
            phrase = " ".join([t.text for t in sorted(phrase_tokens, key=lambda x: x.i)])
            if len(phrase) > 8:  # Skip very short phrases
                concepts.append(phrase.lower().strip())
    
    # Also add the full segment text as a concept (normalized)
    # This catches repeated statements like "I went to the store"
    normalized_text = " ".join(text.lower().split())
    if len(normalized_text) > 10:
        concepts.append(normalized_text)
    
    # Extract sentences (for question repetition)
    sentences = [sent.text.strip() for sent in doc.sents]
    
    return concepts, sentences


def similarity_score(s1: str, s2: str, use_semantic: bool = True) -> float:
    """
    Calculate similarity between two strings.
    
    Args:
        s1, s2: Strings to compare
        use_semantic: If True, use semantic similarity (embeddings). 
                      If False, use string matching (faster but less accurate).
    """
    if use_semantic:
        return semantic_similarity(s1, s2)
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def time_decay_score(time_gap_seconds: float) -> float:
    """
    Calculate a concern score based on time gap between repetitions.
    
    Longer gaps are MORE concerning for dementia screening:
    - < 10 seconds: likely self-correction (0.1)
    - 10-30 seconds: minor concern (0.3)
    - 30-60 seconds: moderate concern (0.5)
    - 1-5 minutes: significant concern (0.7)
    - 5-15 minutes: high concern (0.85)
    - > 15 minutes: very high concern (1.0)
    
    Returns: score 0.0-1.0 (higher = more concerning)
    """
    if time_gap_seconds < 10:
        return 0.1  # Likely self-correction
    elif time_gap_seconds < 30:
        return 0.3  # Minor
    elif time_gap_seconds < 60:
        return 0.5  # Moderate
    elif time_gap_seconds < 300:  # 5 minutes
        return 0.7  # Significant
    elif time_gap_seconds < 900:  # 15 minutes
        return 0.85  # High
    else:
        return 1.0  # Very high - repeating after 15+ minutes


def calculate_concern_score(similarity: float, time_gap_seconds: float) -> float:
    """
    Calculate overall concern score combining similarity and time decay.
    
    Formula: concern = similarity * (0.5 + 0.5 * time_decay)
    
    This means:
    - High similarity + long gap = high concern
    - High similarity + short gap = moderate concern (might be self-correction)
    - Low similarity isn't flagged regardless of time
    
    Returns: score 0.0-1.0
    """
    time_weight = time_decay_score(time_gap_seconds)
    # Weight: 50% base similarity + 50% time-adjusted
    concern = similarity * (0.5 + 0.5 * time_weight)
    return round(concern, 3)


def detect_repetitions(
    segments: List[Dict[str, Any]], 
    threshold: float = 0.70,  # Lowered for semantic matching
    min_segment_gap: int = 1,
    use_semantic: bool = True,
    min_concern_score: float = 0.0  # Filter by concern score (0 = show all)
) -> Dict[str, Any]:
    """
    Detect repeated concepts and questions - SAME SPEAKER ONLY.
    
    Uses semantic similarity (sentence embeddings) to catch paraphrased repetitions:
    - "I went to the store" ≈ "I visited the shop" 
    - "What time is it?" ≈ "Do you know the time?"
    
    Time decay weighting: repetitions after longer gaps are MORE concerning.
    - < 10s: likely self-correction (low concern)
    - 1-5 min: significant concern
    - > 15 min: very high concern
    
    For dementia screening, we only care about a person repeating themselves,
    not normal conversational echoing between speakers.
    
    Args:
        segments: List of dicts with 'text', 'start' (seconds), and optionally 'speaker'
        threshold: Similarity threshold (0.70 for semantic, 0.80 for string)
        min_segment_gap: Minimum segments apart to count as repetition (1 = adjacent OK)
        use_semantic: Use semantic similarity (True) or string matching (False)
        min_concern_score: Minimum concern score to include (0.0-1.0)
    
    Returns:
        Dict with repetition analysis including concern scores
    """
    all_concepts = []
    all_sentences = []
    
    for i, seg in enumerate(segments):
        text = seg.get('text', '')
        start_time = seg.get('start', 0)
        concepts, sentences = extract_concepts(text)
        
        for c in concepts:
            if len(c) > 5:  # Skip short phrases
                all_concepts.append({
                    'concept': c,
                    'segment_index': i,
                    'start_time': start_time,
                    'speaker': seg.get('speaker', 'unknown')
                })
        
        for s in sentences:
            if len(s) > 15:  # Skip short sentences
                is_question = s.strip().endswith('?')
                all_sentences.append({
                    'sentence': s,
                    'segment_index': i,
                    'start_time': start_time,
                    'speaker': seg.get('speaker', 'unknown'),
                    'is_question': is_question
                })
    
    # Use batch semantic similarity for efficiency
    concept_texts = [c['concept'] for c in all_concepts]
    if use_semantic and len(concept_texts) > 0:
        concept_sim_matrix = batch_semantic_similarity(concept_texts)
    else:
        concept_sim_matrix = None
    
    # Find repeated concepts - SAME SPEAKER ONLY
    repeated_concepts = []
    seen_pairs = set()
    
    for i, c1 in enumerate(all_concepts):
        for j, c2 in enumerate(all_concepts):
            if i >= j:
                continue
            if (i, j) in seen_pairs:
                continue
            
            # SAME SPEAKER CHECK - only count if same person repeating
            if c1['speaker'] != c2['speaker']:
                continue
            
            # MINIMUM GAP CHECK - not immediate self-correction
            if abs(c1['segment_index'] - c2['segment_index']) < min_segment_gap:
                continue
            
            # Get similarity from precomputed matrix or compute on-the-fly
            if concept_sim_matrix is not None:
                sim = float(concept_sim_matrix[i, j])
            else:
                sim = similarity_score(c1['concept'], c2['concept'], use_semantic=False)
            
            if sim >= threshold:
                seen_pairs.add((i, j))
                time_gap = abs(c1['start_time'] - c2['start_time'])
                concern = calculate_concern_score(sim, time_gap)
                
                # Filter by minimum concern score
                if concern >= min_concern_score:
                    repeated_concepts.append({
                        'concept_1': c1['concept'],
                        'concept_2': c2['concept'],
                        'similarity': round(sim, 2),
                        'time_gap_seconds': round(time_gap, 1),
                        'time_decay_weight': time_decay_score(time_gap),
                        'concern_score': concern,
                        'speaker': c1['speaker'],
                        'segments': [c1['segment_index'], c2['segment_index']]
                    })
    
    # Batch compute question similarities
    questions = [s for s in all_sentences if s['is_question']]
    question_texts = [q['sentence'] for q in questions]
    if use_semantic and len(question_texts) > 0:
        question_sim_matrix = batch_semantic_similarity(question_texts)
    else:
        question_sim_matrix = None
    
    # Find repeated questions - SAME SPEAKER ONLY
    repeated_questions = []
    question_pairs = set()
    for i, q1 in enumerate(questions):
        for j, q2 in enumerate(questions):
            if i >= j:
                continue
            if (i, j) in question_pairs:
                continue
            
            # SAME SPEAKER CHECK
            if q1['speaker'] != q2['speaker']:
                continue
            
            # MINIMUM GAP CHECK
            if abs(q1['segment_index'] - q2['segment_index']) < min_segment_gap:
                continue
            
            # Get similarity from precomputed matrix or compute on-the-fly
            if question_sim_matrix is not None:
                sim = float(question_sim_matrix[i, j])
            else:
                sim = similarity_score(q1['sentence'], q2['sentence'], use_semantic=False)
            
            if sim >= threshold:
                question_pairs.add((i, j))
                time_gap = abs(q1['start_time'] - q2['start_time'])
                concern = calculate_concern_score(sim, time_gap)
                
                # Filter by minimum concern score
                if concern >= min_concern_score:
                    repeated_questions.append({
                        'question_1': q1['sentence'],
                        'question_2': q2['sentence'],
                        'similarity': round(sim, 2),
                        'time_gap_seconds': round(time_gap, 1),
                        'time_decay_weight': time_decay_score(time_gap),
                        'concern_score': concern,
                        'speaker': q1['speaker'],
                        'segments': [q1['segment_index'], q2['segment_index']]
                    })
    
    # Sort by concern score (most concerning first)
    repeated_concepts.sort(key=lambda x: x['concern_score'], reverse=True)
    repeated_questions.sort(key=lambda x: x['concern_score'], reverse=True)
    
    # Calculate weighted concern - average of top concerns
    all_concerns = [c['concern_score'] for c in repeated_concepts + repeated_questions]
    avg_concern = sum(all_concerns) / len(all_concerns) if all_concerns else 0.0
    max_concern = max(all_concerns) if all_concerns else 0.0
    
    # Calculate total repetition count for storage
    total_repetition_count = len(repeated_concepts) + len(repeated_questions)
    
    return {
        'repeated_concepts_count': len(repeated_concepts),
        'repeated_questions_count': len(repeated_questions),
        'total_repetition_count': total_repetition_count,
        'average_concern_score': round(avg_concern, 3),
        'max_concern_score': round(max_concern, 3),
        'repeated_concepts': repeated_concepts[:20],  # Limit to top 20
        'repeated_questions': repeated_questions[:20],
        'total_concepts_analyzed': len(all_concepts),
        'total_questions_analyzed': len(questions),
        'semantic_matching': use_semantic
    }


def detect_aphasic_events(text: str) -> Dict[str, Any]:
    """
    Detect word-finding difficulties (aphasic events) in speech.
    
    Indicators:
    1. Filler words/hesitations (um, uh, er, ah)
    2. Word-searching phrases ("what's it called", "you know", "the thing")
    3. Self-corrections and restarts ("I went to the... the store")
    4. Circumlocution patterns ("the thing you use to write" instead of "pen")
    5. Trailing off / incomplete thoughts
    
    Returns confidence score and list of detected events WITH character positions for highlighting.
    """
    import re
    text_lower = text.lower()
    events = []
    highlights = []  # For transcript highlighting
    
    # 1. Filler/hesitation words (weighted by frequency) - WITH POSITIONS
    fillers = {
        'um': 0.3, 'uh': 0.3, 'er': 0.3, 'ah': 0.2, 'umm': 0.3, 'uhh': 0.3,
        'hmm': 0.2, 'hm': 0.2, 'eh': 0.2
    }
    for filler, weight in fillers.items():
        for match in re.finditer(rf'\b{filler}\b', text_lower):
            events.append({
                'type': 'hesitation',
                'marker': filler,
                'confidence': 0.6 + weight,
                'start': match.start(),
                'end': match.end(),
            })
            highlights.append({
                'type': 'hesitation',
                'start': match.start(),
                'end': match.end(),
                'text': text[match.start():match.end()],
                'color': 'yellow',  # Hesitations in yellow
            })
    
    # 2. Word-searching phrases - WITH POSITIONS
    search_phrases = [
        (r"what'?s it called", 0.9),
        (r"what do you call it", 0.9),
        (r"you know[,.]* the", 0.7),
        (r"the thing[,.]* (the |that |you )", 0.8),
        (r"what'?s the word", 0.95),
        (r"i can'?t (think of|remember) the (word|name)", 0.95),
        (r"it'?s on the tip of my tongue", 0.95),
        (r"how do you say", 0.7),
        (r"that (thing|stuff|place)", 0.5),
        (r"whatchamacallit", 0.9),
        (r"thingamajig", 0.9),
        (r"what'?s (his|her) name", 0.6),
    ]
    
    for pattern, conf in search_phrases:
        for match in re.finditer(pattern, text_lower):
            events.append({
                'type': 'word_search',
                'marker': text[match.start():match.end()],
                'confidence': conf,
                'start': match.start(),
                'end': match.end(),
            })
            highlights.append({
                'type': 'word_search',
                'start': match.start(),
                'end': match.end(),
                'text': text[match.start():match.end()],
                'color': 'orange',  # Word searches in orange
            })
    
    # 3. Self-corrections / restarts - WITH POSITIONS
    restart_pattern = r'\b(\w+)[,.\s]+\1\b'
    for match in re.finditer(restart_pattern, text_lower):
        word = match.group(1)
        if word in ['the', 'a', 'i'] or word not in ['and', 'to', 'that', 'very']:
            conf = 0.8 if word in ['the', 'a', 'i'] else 0.7
            events.append({
                'type': 'restart',
                'marker': f'{word} {word}',
                'confidence': conf,
                'start': match.start(),
                'end': match.end(),
            })
            highlights.append({
                'type': 'restart',
                'start': match.start(),
                'end': match.end(),
                'text': text[match.start():match.end()],
                'color': 'red',  # Restarts in red
            })
    
    # 4. Incomplete/trailing sentences - WITH POSITIONS
    for match in re.finditer(r'[^.!?]+\.{3}', text):
        events.append({
            'type': 'incomplete',
            'marker': match.group()[-30:] + '...',
            'confidence': 0.75,
            'start': match.start(),
            'end': match.end(),
        })
        highlights.append({
            'type': 'incomplete',
            'start': match.start(),
            'end': match.end(),
            'text': text[match.start():match.end()],
            'color': 'purple',  # Incomplete thoughts in purple
        })
    
    # 5. Long pauses - WITH POSITIONS
    for match in re.finditer(r'\[\.+\]|\(pause\)|\(long pause\)|\.\.\.\s*\.\.\.', text_lower):
        events.append({
            'type': 'pause',
            'marker': 'extended pause',
            'confidence': 0.7,
            'start': match.start(),
            'end': match.end(),
        })
        highlights.append({
            'type': 'pause',
            'start': match.start(),
            'end': match.end(),
            'text': text[match.start():match.end()],
            'color': 'gray',  # Pauses in gray
        })
    
    # Calculate overall aphasic score
    # Weight events by confidence and normalize by word count
    doc = nlp(text)
    word_count = len([t for t in doc if t.is_alpha])
    
    if word_count == 0:
        return {
            'event_count': 0,
            'events_per_100_words': 0.0,
            'confidence_score': 0.0,
            'events': [],
            'word_count': 0
        }
    
    total_confidence = sum(e['confidence'] for e in events)
    events_per_100 = (len(events) / word_count) * 100 if word_count > 0 else 0
    
    # Confidence score: 0-1 scale based on event density and confidence
    # More events + higher confidence = higher score
    confidence_score = min(1.0, (total_confidence / max(word_count / 20, 1)))
    
    # Sort highlights by position for proper rendering
    highlights.sort(key=lambda x: x['start'])
    
    return {
        'event_count': len(events),
        'events_per_100_words': round(events_per_100, 2),
        'confidence_score': round(confidence_score, 2),
        'events': events[:15],  # Limit to top 15 examples
        'highlights': highlights,  # All highlights with positions for transcript coloring
        'word_count': word_count,
        'breakdown': {
            'hesitations': len([e for e in events if e['type'] == 'hesitation']),
            'word_searches': len([e for e in events if e['type'] == 'word_search']),
            'restarts': len([e for e in events if e['type'] == 'restart']),
            'incomplete': len([e for e in events if e['type'] == 'incomplete']),
            'pauses': len([e for e in events if e['type'] == 'pause']),
        }
    }


def analyze_dementia_markers(
    segments: List[Dict[str, Any]],
    speaker_labels: List[str] = None,
    repetition_threshold: float = 0.85,
    min_segment_gap: int = 2,
    user_id: Optional[int] = None,
    speech_id: Optional[int] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full dementia marker analysis for a conversation.
    
    Args:
        segments: List of speech segments with 'text', 'speaker', 'start', 'end'
        speaker_labels: Optional friendly names for speakers
        repetition_threshold: Similarity threshold for repetition detection (0.5-1.0)
        min_segment_gap: Minimum segments apart to count as repetition
        user_id: User ID for cross-session tracking (optional)
        speech_id: Current speech ID for cross-session tracking (optional)
        db_path: Path to database for cross-session tracking (optional)
    
    Returns:
        Complete analysis with per-speaker metrics and cross-session data
    """
    if not speaker_labels:
        # Auto-detect speakers from segments
        speakers = list(set(seg.get('speaker', 'Speaker 1') for seg in segments))
        speaker_labels = {s: s for s in speakers}
    
    # Group segments by speaker
    speaker_segments = {}
    for seg in segments:
        speaker = seg.get('speaker', 'Speaker 1')
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
        speaker_segments[speaker].append(seg)
    
    # Analyze each speaker
    speaker_results = {}
    for speaker, segs in speaker_segments.items():
        # Combine all text for this speaker
        full_text = ' '.join(seg.get('text', '') for seg in segs)
        
        # Preposition analysis
        prep_analysis = calculate_preposition_rate(full_text)
        
        # Repetition analysis (just for this speaker's segments)
        rep_analysis = detect_repetitions(segs, threshold=repetition_threshold, min_segment_gap=min_segment_gap)
        
        # Aphasic event detection (word-finding difficulties)
        aphasic_analysis = detect_aphasic_events(full_text)
        
        speaker_results[speaker] = {
            'label': speaker_labels.get(speaker, speaker),
            'total_segments': len(segs),
            'total_words': prep_analysis['word_count'],
            'preposition_analysis': prep_analysis,
            'repetition_analysis': rep_analysis,
            'aphasic_analysis': aphasic_analysis,
            'metrics': {
                'prepositions_per_10_words': prep_analysis['rate_per_10_words'],
                'repeated_concepts': rep_analysis['repeated_concepts_count'],
                'repeated_questions': rep_analysis['repeated_questions_count'],
                'aphasic_events': aphasic_analysis['event_count'],
                'aphasic_confidence': aphasic_analysis['confidence_score']
            }
        }
    
    # Cross-speaker repetition (if one speaker repeats what the other said)
    cross_repetition = detect_repetitions(segments, threshold=0.6)
    
    # Risk assessment based on metrics
    # Normal preposition rate: ~1.0-1.5 per 10 words
    # Concerning: < 0.7 per 10 words
    risk_flags = []
    for speaker, data in speaker_results.items():
        prep_rate = data['metrics']['prepositions_per_10_words']
        rep_count = data['metrics']['repeated_concepts'] + data['metrics']['repeated_questions']
        aphasic_count = data['metrics']['aphasic_events']
        aphasic_conf = data['metrics']['aphasic_confidence']
        
        if prep_rate < 0.7 and data['total_words'] > 50:
            risk_flags.append({
                'speaker': speaker,
                'flag': 'low_preposition_rate',
                'value': prep_rate,
                'concern': 'Preposition usage below typical range (< 0.7 per 10 words)'
            })
        
        if rep_count >= 3:
            risk_flags.append({
                'speaker': speaker,
                'flag': 'high_repetition',
                'value': rep_count,
                'concern': f'Multiple repeated concepts/questions detected ({rep_count})'
            })
        
        # Aphasic events: flag if multiple events with decent confidence
        if aphasic_count >= 3 and aphasic_conf >= 0.3:
            risk_flags.append({
                'speaker': speaker,
                'flag': 'word_finding_difficulty',
                'value': aphasic_count,
                'confidence': aphasic_conf,
                'concern': f'Word-finding difficulties detected ({aphasic_count} events, {int(aphasic_conf*100)}% confidence)'
            })
        elif aphasic_count >= 5:  # Even low confidence matters if frequent
            risk_flags.append({
                'speaker': speaker,
                'flag': 'word_finding_difficulty',
                'value': aphasic_count,
                'confidence': aphasic_conf,
                'concern': f'Frequent hesitations/word-searching ({aphasic_count} events)'
            })
    
    # Cross-session tracking (if user_id provided)
    cross_session_analysis = None
    if user_id is not None and db_path is not None:
        try:
            from .cross_session_tracking import (
                detect_cross_session_repetitions,
                store_concepts,
                get_embedding
            )
            
            # Extract all concepts and questions from segments
            all_concepts = []
            all_questions = []
            concepts_to_store = []
            
            for seg in segments:
                text = seg.get('text', '')
                concepts, sentences = extract_concepts(text)
                all_concepts.extend([c for c in concepts if len(c) > 10])
                
                # Identify questions
                for sent in sentences:
                    if sent.strip().endswith('?'):
                        all_questions.append(sent)
                        concepts_to_store.append({
                            'text': sent,
                            'type': 'question'
                        })
                    elif len(sent) > 15:
                        concepts_to_store.append({
                            'text': sent,
                            'type': 'statement'
                        })
            
            # Detect cross-session repetitions
            cross_session_analysis = detect_cross_session_repetitions(
                db_path=db_path,
                user_id=user_id,
                current_concepts=all_concepts[:50],  # Limit to avoid overload
                current_questions=all_questions[:20],
                similarity_threshold=0.70
            )
            
            # Store concepts for future tracking (if speech_id provided)
            if speech_id is not None and concepts_to_store:
                store_result = store_concepts(
                    db_path=db_path,
                    user_id=user_id,
                    speech_id=speech_id,
                    concepts=concepts_to_store[:100]  # Limit storage
                )
                cross_session_analysis['storage'] = store_result
            
            # Add cross-session risk flags
            if cross_session_analysis.get('max_concern_level', 0) >= 0.7:
                risk_flags.append({
                    'speaker': 'cross_session',
                    'flag': 'cross_session_repetition',
                    'value': cross_session_analysis['total_cross_session_matches'],
                    'concern': f"Repeating concepts from previous sessions ({cross_session_analysis['total_cross_session_matches']} matches, max concern: {cross_session_analysis['max_concern_level']:.0%})"
                })
            
        except Exception as e:
            logger.warning(f"Cross-session tracking failed: {e}")
            cross_session_analysis = {'enabled': False, 'error': str(e)}
    
    return {
        'speakers': speaker_results,
        'cross_speaker_repetition': cross_repetition,
        'cross_session_analysis': cross_session_analysis,
        'risk_flags': risk_flags,
        'summary': {
            'total_speakers': len(speaker_results),
            'total_segments': len(segments),
            'flags_detected': len(risk_flags),
            'cross_session_enabled': cross_session_analysis is not None
        }
    }
