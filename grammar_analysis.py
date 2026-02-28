"""
Grammar Analysis Module
Detects filler words, grammar issues, and provides improvement suggestions.
"""

import re
from typing import List, Dict, Any
from dataclasses import dataclass


# Common filler words/phrases
FILLER_WORDS = {
    "um", "uh", "er", "ah", "like", "you know", "basically", "actually",
    "literally", "honestly", "right", "okay", "so", "well", "i mean",
    "kind of", "sort of", "you see", "i guess", "whatever", "anyway",
}

# Patterns that might indicate grammar issues
GRAMMAR_PATTERNS = [
    # Double words
    (r'\b(\w+)\s+\1\b', 'repeated_word', 'Remove repeated word'),
    # "Me and X" vs "X and I"
    (r'\bme and\b', 'pronoun_order', 'Consider "... and I" for subject position'),
    # Double negatives
    (r"\bdon't\s+(?:got\s+)?no\b|\bain't\s+no\b|\bcan't\s+(?:get\s+)?no\b", 'double_negative', 'Avoid double negatives'),
    # "Could of" instead of "could have"
    (r'\b(?:could|would|should|must|might)\s+of\b', 'of_vs_have', 'Use "have" instead of "of"'),
    # Run-on "and then and then"
    (r'\band then\s+and then\b', 'run_on', 'Simplify repeated connectors'),
]

# Weak/vague words that could be strengthened
WEAK_WORDS = {
    "thing": "Consider being more specific",
    "stuff": "Consider being more specific", 
    "things": "Consider being more specific",
    "very": "Consider a stronger adjective",
    "really": "Consider a stronger adverb",
    "a lot": "Consider quantifying or using 'many/much'",
    "got": "Consider 'received', 'obtained', or 'have'",
    "nice": "Consider a more descriptive adjective",
    "good": "Consider a more descriptive adjective",
    "bad": "Consider a more descriptive adjective",
}

# Hedging/confidence-undermining phrases
# These phrases weaken authority and reduce perceived confidence
HEDGING_PHRASES = {
    # Uncertainty markers
    "i think": "State directly without hedging",
    "i believe": "State directly without hedging",
    "i feel like": "State directly without hedging",
    "i guess": "State directly without hedging",
    "maybe": "Be more definitive",
    "perhaps": "Be more definitive",
    "probably": "Be more definitive",
    "possibly": "Be more definitive",
    # Qualifiers
    "kind of": "Remove qualifier for stronger statement",
    "sort of": "Remove qualifier for stronger statement",
    "somewhat": "Remove qualifier for stronger statement",
    "a little bit": "Remove qualifier for stronger statement",
    "just": "Often unnecessary - try removing",
    # Permission-seeking
    "if that makes sense": "Trust your explanation",
    "does that make sense": "Trust your explanation",
    "if you know what i mean": "Trust your explanation",
    # Apologetic
    "sorry but": "No need to apologize for your point",
    "i'm sorry but": "No need to apologize for your point",
    "i'm no expert but": "Own your expertise",
    "this might be wrong but": "State confidently or verify first",
    # Minimizers
    "i was just going to say": "Just say it directly",
    "i just wanted to": "Just do/say it directly",
}


@dataclass
class GrammarIssue:
    """Represents a grammar or style issue in the transcript."""
    type: str  # filler, grammar, weak_word, suggestion
    text: str  # The problematic text
    position: int  # Character position in transcript
    length: int  # Length of the issue span
    severity: str  # info, warning, error
    suggestion: str  # Improvement suggestion


def analyze_grammar(transcript: str) -> Dict[str, Any]:
    """
    Analyze transcript for grammar issues, filler words, and style improvements.
    
    Returns:
        dict with:
        - issues: List of GrammarIssue objects as dicts
        - filler_count: Total filler words detected
        - filler_ratio: Ratio of filler words to total words
        - suggestions: List of general improvement suggestions
        - word_count: Total word count
    """
    if not transcript:
        return {
            "issues": [],
            "filler_count": 0,
            "filler_ratio": 0.0,
            "suggestions": [],
            "word_count": 0,
        }
    
    transcript_lower = transcript.lower()
    words = transcript_lower.split()
    word_count = len(words)
    
    issues = []
    filler_count = 0
    
    # Detect filler words
    for filler in FILLER_WORDS:
        # Use word boundary for single words, looser match for phrases
        if ' ' in filler:
            pattern = re.escape(filler)
        else:
            pattern = r'\b' + re.escape(filler) + r'\b'
        
        for match in re.finditer(pattern, transcript_lower):
            filler_count += 1
            issues.append({
                "type": "filler",
                "text": match.group(),
                "position": match.start(),
                "length": len(match.group()),
                "severity": "info",
                "suggestion": f"Consider removing filler word '{match.group()}'",
            })
    
    # Detect grammar patterns
    for pattern, issue_type, suggestion in GRAMMAR_PATTERNS:
        for match in re.finditer(pattern, transcript_lower, re.IGNORECASE):
            issues.append({
                "type": issue_type,
                "text": match.group(),
                "position": match.start(),
                "length": len(match.group()),
                "severity": "warning",
                "suggestion": suggestion,
            })
    
    # Detect weak words
    for weak_word, suggestion in WEAK_WORDS.items():
        if ' ' in weak_word:
            pattern = re.escape(weak_word)
        else:
            pattern = r'\b' + re.escape(weak_word) + r'\b'
        
        for match in re.finditer(pattern, transcript_lower):
            issues.append({
                "type": "weak_word",
                "text": match.group(),
                "position": match.start(),
                "length": len(match.group()),
                "severity": "info",
                "suggestion": suggestion,
            })
    
    # Detect hedging/confidence-undermining phrases
    hedging_count = 0
    for hedge, suggestion in HEDGING_PHRASES.items():
        if ' ' in hedge:
            pattern = re.escape(hedge)
        else:
            pattern = r'\b' + re.escape(hedge) + r'\b'
        
        for match in re.finditer(pattern, transcript_lower):
            hedging_count += 1
            issues.append({
                "type": "hedging",
                "text": match.group(),
                "position": match.start(),
                "length": len(match.group()),
                "severity": "warning",
                "suggestion": suggestion,
            })
    
    # Sort issues by position
    issues.sort(key=lambda x: x["position"])
    
    # Calculate filler ratio
    filler_ratio = filler_count / word_count if word_count > 0 else 0.0
    
    # Generate overall suggestions
    suggestions = []
    
    if filler_ratio > 0.05:
        suggestions.append({
            "type": "high_filler_rate",
            "message": f"High filler word rate ({filler_ratio:.1%}). Practice pausing instead of using filler words.",
        })
    elif filler_ratio > 0.02:
        suggestions.append({
            "type": "moderate_filler_rate", 
            "message": f"Moderate filler word rate ({filler_ratio:.1%}). Some fillers are natural, but reducing them can improve clarity.",
        })
    
    # Check for repeated word patterns indicating hesitation
    repeated_found = any(i["type"] == "repeated_word" for i in issues)
    if repeated_found:
        suggestions.append({
            "type": "repeated_words",
            "message": "Repeated words detected. This often indicates hesitation - try pausing to collect thoughts.",
        })
    
    # Calculate hedging ratio
    hedging_ratio = hedging_count / word_count if word_count > 0 else 0.0
    
    # Calculate confidence score (0-100)
    # Start at 100, deduct for hedging and weak words
    # Target: <2% hedging phrases for score >80
    weak_word_count = sum(1 for i in issues if i["type"] == "weak_word")
    
    # Hedging has bigger impact than weak words
    hedging_penalty = min(hedging_count * 5, 40)  # Max 40 point penalty
    weak_word_penalty = min(weak_word_count * 2, 20)  # Max 20 point penalty
    filler_penalty = min(filler_count * 1.5, 20)  # Max 20 point penalty
    
    confidence_score = max(0, 100 - hedging_penalty - weak_word_penalty - filler_penalty)
    
    # Add hedging-specific suggestion
    if hedging_ratio > 0.03:
        suggestions.append({
            "type": "high_hedging",
            "message": f"High hedging rate ({hedging_count} instances). Own your statements - speak with authority.",
        })
    elif hedging_ratio > 0.01:
        suggestions.append({
            "type": "moderate_hedging",
            "message": f"Some hedging detected ({hedging_count} instances). Consider being more direct.",
        })
    
    return {
        "issues": issues,
        "filler_count": filler_count,
        "filler_ratio": round(filler_ratio, 4),
        "hedging_count": hedging_count,
        "hedging_ratio": round(hedging_ratio, 4),
        "confidence_score": round(confidence_score, 1),
        "suggestions": suggestions,
        "word_count": word_count,
    }


def get_annotated_transcript(transcript: str, grammar_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create an annotated transcript with grammar issues marked.
    
    Returns list of segments with:
    - text: The text content
    - type: "normal", "filler", "grammar", "weak_word"
    - issue: The issue dict if applicable
    """
    if not transcript or not grammar_result.get("issues"):
        return [{"text": transcript, "type": "normal", "issue": None}]
    
    issues = grammar_result["issues"]
    segments = []
    last_end = 0
    
    for issue in issues:
        pos = issue["position"]
        length = issue["length"]
        
        # Add normal text before this issue
        if pos > last_end:
            segments.append({
                "text": transcript[last_end:pos],
                "type": "normal",
                "issue": None,
            })
        
        # Add the issue segment
        segments.append({
            "text": transcript[pos:pos + length],
            "type": issue["type"],
            "issue": issue,
        })
        
        last_end = pos + length
    
    # Add remaining text
    if last_end < len(transcript):
        segments.append({
            "text": transcript[last_end:],
            "type": "normal",
            "issue": None,
        })
    
    return segments


if __name__ == "__main__":
    # Test
    test = """Um, so basically what I want to say is like, you know, 
    this thing is really very good. I mean, me and my friend got a lot of stuff 
    done and and we could of done more but whatever."""
    
    result = analyze_grammar(test)
    print(f"Word count: {result['word_count']}")
    print(f"Filler count: {result['filler_count']}")
    print(f"Filler ratio: {result['filler_ratio']:.1%}")
    print(f"\nIssues ({len(result['issues'])}):")
    for issue in result['issues']:
        print(f"  [{issue['severity']}] {issue['type']}: '{issue['text']}' - {issue['suggestion']}")
    print(f"\nSuggestions:")
    for s in result['suggestions']:
        print(f"  - {s['message']}")
