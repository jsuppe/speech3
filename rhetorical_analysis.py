"""
Rhetorical analysis for the Melchior Voice Pipeline.
Detects anaphora, repetition patterns, and prosodic rhythm.
"""
import re
import numpy as np
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize


def analyze_repetition(transcript, segments=None):
    """
    Detect rhetorical repetition patterns:
    - Anaphora: repeated sentence beginnings ("I have a dream...", "Let freedom ring...")
    - Epistrophe: repeated sentence endings
    - Key phrase repetition: phrases repeated 3+ times
    - Lexical density adjusted for intentional repetition
    """
    sentences = sent_tokenize(transcript)
    if len(sentences) < 3:
        return {
            "anaphora": [],
            "epistrophe": [],
            "repeated_phrases": [],
            "repetition_score": 0,
            "rhetorical_repetition": False,
        }

    # --- Anaphora detection (repeated beginnings) ---
    beginnings = []
    for sent in sentences:
        words = word_tokenize(sent.lower())
        if len(words) >= 3:
            beginnings.append(" ".join(words[:3]))
        elif len(words) >= 2:
            beginnings.append(" ".join(words[:2]))

    beginning_counts = Counter(beginnings)
    anaphora = [
        {"phrase": phrase, "count": count}
        for phrase, count in beginning_counts.most_common()
        if count >= 2
    ]

    # --- Epistrophe detection (repeated endings) ---
    endings = []
    for sent in sentences:
        words = word_tokenize(sent.lower())
        # Strip trailing punctuation tokens
        words = [w for w in words if re.match(r'\w', w)]
        if len(words) >= 3:
            endings.append(" ".join(words[-3:]))
        elif len(words) >= 2:
            endings.append(" ".join(words[-2:]))

    ending_counts = Counter(endings)
    epistrophe = [
        {"phrase": phrase, "count": count}
        for phrase, count in ending_counts.most_common()
        if count >= 2
    ]

    # --- Repeated phrases (n-grams, n=3-6) ---
    words = word_tokenize(transcript.lower())
    words = [w for w in words if re.match(r'\w', w)]

    repeated_phrases = []
    for n in range(3, 7):
        ngrams = [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
        ngram_counts = Counter(ngrams)
        for phrase, count in ngram_counts.most_common():
            if count >= 3:
                # Check it's not a subset of an already-found longer phrase
                if not any(phrase in rp["phrase"] and len(rp["phrase"]) > len(phrase)
                           for rp in repeated_phrases):
                    repeated_phrases.append({"phrase": phrase, "count": count, "n": n})

    # Deduplicate: if a shorter phrase is contained in a longer one with same+ count, drop shorter
    final_phrases = []
    for rp in sorted(repeated_phrases, key=lambda x: -x["n"]):
        if not any(rp["phrase"] in fp["phrase"] for fp in final_phrases):
            final_phrases.append(rp)

    # --- Repetition score ---
    # Higher score = more intentional repetition detected
    anaphora_score = sum(a["count"] for a in anaphora)
    phrase_score = sum(p["count"] * p["n"] for p in final_phrases)
    total_score = anaphora_score + phrase_score

    # Normalize by sentence count
    norm_score = round(total_score / max(len(sentences), 1), 2)

    # Is this likely rhetorical (intentional) repetition?
    rhetorical = (
        any(a["count"] >= 3 for a in anaphora) or
        any(p["count"] >= 3 and p["n"] >= 4 for p in final_phrases) or
        norm_score > 3.0
    )

    return {
        "anaphora": anaphora[:10],
        "epistrophe": epistrophe[:10],
        "repeated_phrases": final_phrases[:10],
        "repetition_score": norm_score,
        "rhetorical_repetition": rhetorical,
        "sentence_count": len(sentences),
    }


def analyze_rhythm(segments):
    """
    Analyze prosodic rhythm from segment timestamps.
    Measures regularity of utterance lengths and pause patterns.
    
    Good speakers have more regular rhythm — predictable cadence
    that audiences can "lock into."
    """
    if not segments or len(segments) < 3:
        return {
            "utterance_duration_mean": 0,
            "utterance_duration_std": 0,
            "rhythm_regularity": 0,
            "pause_regularity": 0,
            "tempo_variation": [],
        }

    # Utterance durations
    durations = [s["end"] - s["start"] for s in segments]
    dur_mean = float(np.mean(durations))
    dur_std = float(np.std(durations))

    # Rhythm regularity: coefficient of variation (lower = more regular)
    # Invert and normalize to 0-1 where 1 = perfectly regular
    cv = dur_std / dur_mean if dur_mean > 0 else 1
    rhythm_regularity = round(max(0, 1 - cv), 3)

    # Pause durations between segments
    pauses = []
    for i in range(len(segments) - 1):
        gap = segments[i + 1]["start"] - segments[i]["end"]
        if gap > 0.1:
            pauses.append(gap)

    if pauses:
        pause_cv = float(np.std(pauses)) / float(np.mean(pauses)) if np.mean(pauses) > 0 else 1
        pause_regularity = round(max(0, 1 - pause_cv), 3)
    else:
        pause_regularity = 0

    # Tempo variation over time (WPM per segment)
    tempo_variation = []
    for s in segments:
        dur = s["end"] - s["start"]
        if dur > 0:
            words = len(s["text"].split())
            wpm = (words / dur) * 60
            tempo_variation.append({
                "start": round(s["start"], 2),
                "wpm": round(float(wpm), 1)
            })

    return {
        "utterance_duration_mean": round(dur_mean, 2),
        "utterance_duration_std": round(dur_std, 2),
        "rhythm_regularity": rhythm_regularity,
        "pause_regularity": pause_regularity,
        "tempo_variation": tempo_variation,
    }


def analyze_rhetoric(transcript, segments=None):
    """Full rhetorical analysis combining repetition and rhythm."""
    repetition = analyze_repetition(transcript, segments)
    rhythm = analyze_rhythm(segments) if segments else {}

    return {
        "repetition": repetition,
        "rhythm": rhythm,
    }
