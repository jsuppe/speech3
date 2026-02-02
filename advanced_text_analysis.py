"""
Advanced text analysis for the Melchior Voice Pipeline.
Readability, discourse structure, question density, named entities,
and perspective analysis.
"""
import re
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy

nlp = spacy.load('en_core_web_sm')


def analyze_readability(transcript):
    """
    Compute readability metrics adapted for spoken text.
    - Flesch Reading Ease (higher = easier)
    - Flesch-Kincaid Grade Level
    - Average syllables per word
    """
    words = word_tokenize(transcript)
    sentences = sent_tokenize(transcript)
    
    if not words or not sentences:
        return {"flesch_ease": 0, "fk_grade": 0, "avg_syllables": 0}
    
    # Simple syllable counter
    def count_syllables(word):
        word = word.lower().strip(".,!?;:\"'")
        if not word:
            return 0
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
            count += 1
        for i in range(1, len(word)):
            if word[i] in vowels and word[i-1] not in vowels:
                count += 1
        if word.endswith("e") and count > 1:
            count -= 1
        return max(count, 1)
    
    total_syllables = sum(count_syllables(w) for w in words if w.isalpha())
    word_count = len([w for w in words if w.isalpha()])
    sent_count = len(sentences)
    
    if word_count == 0 or sent_count == 0:
        return {"flesch_ease": 0, "fk_grade": 0, "avg_syllables": 0}
    
    avg_sentence_len = word_count / sent_count
    avg_syllables = total_syllables / word_count
    
    # Flesch Reading Ease
    flesch = 206.835 - (1.015 * avg_sentence_len) - (84.6 * avg_syllables)
    
    # Flesch-Kincaid Grade Level
    fk_grade = (0.39 * avg_sentence_len) + (11.8 * avg_syllables) - 15.59
    
    # Interpret
    if flesch >= 80:
        level = "very easy (6th grade)"
    elif flesch >= 60:
        level = "standard (8th-9th grade)"
    elif flesch >= 40:
        level = "difficult (college level)"
    else:
        level = "very difficult (graduate level)"
    
    return {
        "flesch_reading_ease": round(float(flesch), 1),
        "flesch_kincaid_grade": round(float(fk_grade), 1),
        "avg_syllables_per_word": round(float(avg_syllables), 2),
        "avg_sentence_length": round(float(avg_sentence_len), 1),
        "reading_level": level,
    }


def analyze_discourse(transcript):
    """
    Analyze discourse structure — connectors, transitions, argument markers.
    These indicate how well-organized the speech is.
    """
    text_lower = transcript.lower()
    words = word_tokenize(text_lower)
    word_count = len(words)
    
    # Discourse marker categories
    markers = {
        "causal": ["because", "therefore", "thus", "consequently", "hence", "since", "so"],
        "contrastive": ["but", "however", "although", "yet", "nevertheless", "despite", "whereas", "while"],
        "additive": ["and", "also", "furthermore", "moreover", "additionally", "besides", "too"],
        "temporal": ["then", "next", "finally", "first", "second", "meanwhile", "afterwards", "before", "after"],
        "exemplifying": ["for example", "for instance", "such as", "like", "including"],
        "concluding": ["in conclusion", "to summarize", "overall", "ultimately", "in short"],
    }
    
    marker_counts = {}
    total_markers = 0
    for category, terms in markers.items():
        count = 0
        for term in terms:
            if " " in term:
                count += text_lower.count(term)
            else:
                count += words.count(term)
        marker_counts[category] = count
        total_markers += count
    
    # Discourse density (markers per 100 words)
    density = (total_markers / word_count) * 100 if word_count > 0 else 0
    
    # Dominant discourse mode
    dominant = max(marker_counts, key=marker_counts.get) if marker_counts else "none"
    
    return {
        "marker_counts": marker_counts,
        "total_markers": total_markers,
        "discourse_density": round(float(density), 2),
        "dominant_mode": dominant,
        "structure_quality": (
            "well-structured" if density > 5
            else "moderately structured" if density > 2
            else "loosely structured"
        ),
    }


def analyze_questions(transcript):
    """
    Detect questions — rhetorical questions are a key engagement technique.
    """
    sentences = sent_tokenize(transcript)
    questions = [s for s in sentences if s.strip().endswith("?")]
    
    return {
        "question_count": len(questions),
        "question_ratio": round(len(questions) / max(len(sentences), 1), 3),
        "questions": [q.strip()[:80] for q in questions[:10]],
        "rhetorical_engagement": (
            "high (frequent questions)" if len(questions) > 3
            else "moderate" if len(questions) > 0
            else "none (declarative style)"
        ),
    }


def analyze_named_entities(transcript):
    """
    Named entity analysis — concreteness indicator.
    More named entities = more specific/grounded speech.
    """
    doc = nlp(transcript)
    
    entities = {}
    for ent in doc.ents:
        label = ent.label_
        if label not in entities:
            entities[label] = []
        text = ent.text.strip()
        if text and text not in entities[label]:
            entities[label].append(text)
    
    total_entities = sum(len(v) for v in entities.values())
    word_count = len(word_tokenize(transcript))
    
    return {
        "entity_counts": {k: len(v) for k, v in entities.items()},
        "entities": {k: v[:10] for k, v in entities.items()},
        "total_unique_entities": total_entities,
        "entity_density": round(total_entities / max(word_count, 1) * 100, 2),
        "concreteness": (
            "highly concrete" if total_entities > 10
            else "moderately concrete" if total_entities > 3
            else "abstract"
        ),
    }


def analyze_perspective(transcript):
    """
    Analyze pronoun perspective — first person (I/we) vs second (you) vs third (they/he/she).
    Reveals speaker's stance: personal, inclusive, distanced.
    """
    words = word_tokenize(transcript.lower())
    word_count = len(words)
    
    first_singular = sum(words.count(p) for p in ["i", "me", "my", "mine", "myself"])
    first_plural = sum(words.count(p) for p in ["we", "us", "our", "ours", "ourselves"])
    second = sum(words.count(p) for p in ["you", "your", "yours", "yourself", "yourselves"])
    third = sum(words.count(p) for p in ["he", "she", "they", "him", "her", "them", "his", "their", "its"])
    
    total_pronouns = first_singular + first_plural + second + third
    
    # Dominant perspective
    perspectives = {
        "first_singular": first_singular,
        "first_plural": first_plural,
        "second_person": second,
        "third_person": third,
    }
    dominant = max(perspectives, key=perspectives.get) if total_pronouns > 0 else "none"
    
    # Inclusivity score: "we" usage relative to "I"
    inclusivity = first_plural / max(first_singular + first_plural, 1)
    
    return {
        "first_singular": first_singular,
        "first_plural": first_plural,
        "second_person": second,
        "third_person": third,
        "total_pronouns": total_pronouns,
        "pronoun_density": round(total_pronouns / max(word_count, 1) * 100, 2),
        "dominant_perspective": dominant,
        "inclusivity_ratio": round(float(inclusivity), 3),
        "stance": (
            "inclusive (we-focused)" if inclusivity > 0.5 and first_plural > 3
            else "personal (I-focused)" if first_singular > first_plural * 2
            else "audience-directed (you-focused)" if second > first_singular
            else "narrative (third-person)" if third > first_singular
            else "balanced"
        ),
    }


def analyze_advanced_text(transcript):
    """Full advanced text analysis."""
    return {
        "readability": analyze_readability(transcript),
        "discourse_structure": analyze_discourse(transcript),
        "questions": analyze_questions(transcript),
        "named_entities": analyze_named_entities(transcript),
        "perspective": analyze_perspective(transcript),
    }
