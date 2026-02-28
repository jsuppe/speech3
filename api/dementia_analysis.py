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
from collections import Counter
from typing import List, Dict, Any, Tuple
import spacy
from difflib import SequenceMatcher

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
    
    # Extract sentences (for question repetition)
    sentences = [sent.text.strip() for sent in doc.sents]
    
    return concepts, sentences


def similarity_score(s1: str, s2: str) -> float:
    """Calculate similarity between two strings."""
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def detect_repetitions(segments: List[Dict[str, Any]], threshold: float = 0.85, min_segment_gap: int = 2) -> Dict[str, Any]:
    """
    Detect repeated concepts and questions - SAME SPEAKER ONLY.
    
    For dementia screening, we only care about a person repeating themselves,
    not normal conversational echoing between speakers.
    
    Args:
        segments: List of dicts with 'text' and optionally 'speaker'
        threshold: Similarity threshold (0.85 = 85% similar, higher = stricter)
        min_segment_gap: Minimum segments apart to count as repetition (not immediate self-correction)
    
    Returns:
        Dict with repetition analysis per speaker
    """
    all_concepts = []
    all_sentences = []
    
    for i, seg in enumerate(segments):
        text = seg.get('text', '')
        start_time = seg.get('start', 0)
        concepts, sentences = extract_concepts(text)
        
        for c in concepts:
            if len(c) > 5:  # Skip short phrases (was 3, now 5)
                all_concepts.append({
                    'concept': c,
                    'segment_index': i,
                    'start_time': start_time,
                    'speaker': seg.get('speaker', 'unknown')
                })
        
        for s in sentences:
            if len(s) > 15:  # Skip short sentences (was 10, now 15)
                is_question = s.strip().endswith('?')
                all_sentences.append({
                    'sentence': s,
                    'segment_index': i,
                    'start_time': start_time,
                    'speaker': seg.get('speaker', 'unknown'),
                    'is_question': is_question
                })
    
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
                
            sim = similarity_score(c1['concept'], c2['concept'])
            if sim >= threshold:
                seen_pairs.add((i, j))
                repeated_concepts.append({
                    'concept_1': c1['concept'],
                    'concept_2': c2['concept'],
                    'similarity': round(sim, 2),
                    'speaker': c1['speaker'],
                    'segments': [c1['segment_index'], c2['segment_index']]
                })
    
    # Find repeated questions - SAME SPEAKER ONLY
    repeated_questions = []
    question_pairs = set()
    
    questions = [s for s in all_sentences if s['is_question']]
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
                
            sim = similarity_score(q1['sentence'], q2['sentence'])
            if sim >= threshold:
                question_pairs.add((i, j))
                repeated_questions.append({
                    'question_1': q1['sentence'],
                    'question_2': q2['sentence'],
                    'similarity': round(sim, 2),
                    'speaker': q1['speaker'],
                    'segments': [q1['segment_index'], q2['segment_index']]
                })
    
    # Calculate total repetition count for storage
    total_repetition_count = len(repeated_concepts) + len(repeated_questions)
    
    return {
        'repeated_concepts_count': len(repeated_concepts),
        'repeated_questions_count': len(repeated_questions),
        'total_repetition_count': total_repetition_count,
        'repeated_concepts': repeated_concepts[:20],  # Limit to top 20
        'repeated_questions': repeated_questions[:20],
        'total_concepts_analyzed': len(all_concepts),
        'total_questions_analyzed': len(questions)
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
) -> Dict[str, Any]:
    """
    Full dementia marker analysis for a conversation.
    
    Args:
        segments: List of speech segments with 'text', 'speaker', 'start', 'end'
        speaker_labels: Optional friendly names for speakers
        repetition_threshold: Similarity threshold for repetition detection (0.5-1.0)
        min_segment_gap: Minimum segments apart to count as repetition
    
    Returns:
        Complete analysis with per-speaker metrics
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
    
    return {
        'speakers': speaker_results,
        'cross_speaker_repetition': cross_repetition,
        'risk_flags': risk_flags,
        'summary': {
            'total_speakers': len(speaker_results),
            'total_segments': len(segments),
            'flags_detected': len(risk_flags)
        }
    }
