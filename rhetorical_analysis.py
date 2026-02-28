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


def detect_tricolon(transcript):
    """
    Detect tricolon - groups of three parallel elements.
    Examples:
    - "life, liberty, and the pursuit of happiness"
    - "government of the people, by the people, for the people"
    - "I came, I saw, I conquered"
    """
    tricolons = []
    sentences = sent_tokenize(transcript)
    
    # Pattern 1: "X, Y, and Z" or "X, Y, or Z"
    pattern_and_or = re.compile(
        r'\b([A-Za-z][^,;:.!?]+),\s*([A-Za-z][^,;:.!?]+),\s*(?:and|or)\s+([A-Za-z][^,;:.!?]+)',
        re.IGNORECASE
    )
    
    # Pattern 2: Three parallel phrases with prepositions "of X, of Y, of Z"
    pattern_prep = re.compile(
        r'\b((?:of|by|for|to|with|in)\s+[^,;:.!?]+),\s*((?:of|by|for|to|with|in)\s+[^,;:.!?]+),\s*(?:and\s+)?((?:of|by|for|to|with|in)\s+[^,;:.!?]+)',
        re.IGNORECASE
    )
    
    # Pattern 3: "I X, I Y, I Z" (parallel first-person clauses)
    pattern_parallel_i = re.compile(
        r'\bI\s+(\w+)[^,;:.!?]*,\s*I\s+(\w+)[^,;:.!?]*,\s*I\s+(\w+)',
        re.IGNORECASE
    )
    
    # Pattern 4: Three short clauses separated by commas/semicolons
    pattern_short_clauses = re.compile(
        r'([A-Z][^,;:.!?]{3,30})[,;]\s*([A-Z]?[^,;:.!?]{3,30})[,;]\s*([A-Z]?[^,;:.!?]{3,30})[.!?]',
        re.IGNORECASE
    )
    
    for sent in sentences:
        # Check each pattern
        for match in pattern_and_or.finditer(sent):
            parts = [match.group(1).strip(), match.group(2).strip(), match.group(3).strip()]
            # Check if parts are roughly similar length (parallel structure)
            lengths = [len(p) for p in parts]
            if max(lengths) < min(lengths) * 3:  # Not too imbalanced
                tricolons.append({
                    "type": "list",
                    "parts": parts,
                    "text": match.group(0),
                })
        
        for match in pattern_prep.finditer(sent):
            parts = [match.group(1).strip(), match.group(2).strip(), match.group(3).strip()]
            tricolons.append({
                "type": "prepositional",
                "parts": parts,
                "text": match.group(0),
            })
        
        for match in pattern_parallel_i.finditer(sent):
            verbs = [match.group(1), match.group(2), match.group(3)]
            tricolons.append({
                "type": "parallel_clauses",
                "parts": verbs,
                "text": match.group(0),
            })
    
    # Also check across sentences for three parallel sentence starts
    if len(sentences) >= 3:
        for i in range(len(sentences) - 2):
            words1 = word_tokenize(sentences[i].lower())[:3]
            words2 = word_tokenize(sentences[i+1].lower())[:3]
            words3 = word_tokenize(sentences[i+2].lower())[:3]
            
            if words1 == words2 == words3 and len(words1) >= 2:
                tricolons.append({
                    "type": "sentence_triplet",
                    "parts": [sentences[i][:50], sentences[i+1][:50], sentences[i+2][:50]],
                    "text": " ".join(words1) + "...",
                })
    
    return {
        "tricolons": tricolons[:10],
        "count": len(tricolons),
    }


def detect_rhetorical_questions(transcript):
    """
    Detect rhetorical questions - questions asked for effect, not expecting an answer.
    
    Heuristics for rhetorical questions:
    1. Speaker answers their own question immediately
    2. Multiple questions in succession (question barrage)
    3. Questions with emphatic/emotional words
    4. Questions that are statements in disguise ("Isn't it obvious?")
    """
    sentences = sent_tokenize(transcript)
    questions = []
    rhetorical_questions = []
    
    # Emphatic words that suggest rhetorical intent
    emphatic_words = {
        'how long', 'how many', 'how much', 'how dare',
        'why should', 'why must', 'why do we',
        'what kind', 'what sort', 'what right',
        'who among', 'who here', 'who can',
        'isn\'t it', 'aren\'t we', 'don\'t we', 'won\'t we',
        'can\'t we', 'shouldn\'t we', 'must we',
        'shall we', 'need i', 'dare we',
    }
    
    # Find all questions
    for i, sent in enumerate(sentences):
        if '?' in sent:
            questions.append({
                "text": sent,
                "index": i,
            })
    
    for q in questions:
        sent = q["text"]
        sent_lower = sent.lower()
        idx = q["index"]
        is_rhetorical = False
        reason = ""
        
        # Check 1: Emphatic phrasing
        for phrase in emphatic_words:
            if phrase in sent_lower:
                is_rhetorical = True
                reason = "emphatic phrasing"
                break
        
        # Check 2: Self-answered (speaker answers in next sentence)
        if not is_rhetorical and idx < len(sentences) - 1:
            next_sent = sentences[idx + 1].lower()
            # If next sentence starts with answer-like phrases
            answer_starters = ['it is', 'it\'s', 'because', 'the answer', 'i tell you', 
                              'i say', 'we must', 'we should', 'we will', 'we can']
            for starter in answer_starters:
                if next_sent.startswith(starter):
                    is_rhetorical = True
                    reason = "self-answered"
                    break
        
        # Check 3: Question barrage (3+ questions in a row)
        if not is_rhetorical:
            consecutive = 1
            # Count questions before
            for j in range(idx - 1, -1, -1):
                if '?' in sentences[j]:
                    consecutive += 1
                else:
                    break
            # Count questions after
            for j in range(idx + 1, len(sentences)):
                if '?' in sentences[j]:
                    consecutive += 1
                else:
                    break
            
            if consecutive >= 3:
                is_rhetorical = True
                reason = "question barrage"
        
        # Check 4: Tag questions ("isn't it?", "right?", "don't you think?")
        if not is_rhetorical:
            tag_patterns = [
                r"isn't it\?", r"aren't we\?", r"don't you\?", r"won't we\?",
                r"right\?$", r"correct\?$", r"yes\?$", r"no\?$",
            ]
            for pattern in tag_patterns:
                if re.search(pattern, sent_lower):
                    is_rhetorical = True
                    reason = "tag question"
                    break
        
        # Check 5: Rhetorical patterns ("What greater X than Y?", "Who but X would Y?")
        if not is_rhetorical:
            rhetorical_patterns = [
                r"what (?:greater|better|more|worse)",
                r"who (?:but|else|other than)",
                r"where else",
                r"when (?:if not now|else)",
            ]
            for pattern in rhetorical_patterns:
                if re.search(pattern, sent_lower):
                    is_rhetorical = True
                    reason = "rhetorical pattern"
                    break
        
        if is_rhetorical:
            rhetorical_questions.append({
                "text": sent,
                "reason": reason,
            })
    
    return {
        "total_questions": len(questions),
        "rhetorical_questions": rhetorical_questions[:10],
        "rhetorical_count": len(rhetorical_questions),
        "rhetorical_ratio": round(len(rhetorical_questions) / max(len(questions), 1), 2),
    }


def detect_antithesis(transcript):
    """
    Detect antithesis - contrasting ideas in parallel structure.
    
    Examples:
    - "Ask not what your country can do for you — ask what you can do for your country"
    - "It was the best of times, it was the worst of times"
    - "One small step for man, one giant leap for mankind"
    """
    antitheses = []
    sentences = sent_tokenize(transcript)
    
    # Common antithetical word pairs
    opposites = [
        ('not', 'but'), ('hate', 'love'), ('war', 'peace'), ('rich', 'poor'),
        ('weak', 'strong'), ('enemy', 'friend'), ('death', 'life'), ('dark', 'light'),
        ('old', 'new'), ('past', 'future'), ('worst', 'best'), ('small', 'giant'),
        ('few', 'many'), ('take', 'give'), ('fail', 'succeed'), ('lose', 'win'),
        ('divide', 'unite'), ('destroy', 'create'), ('fear', 'hope'),
        ('despair', 'hope'), ('chains', 'freedom'), ('slave', 'free'),
    ]
    
    # Pattern 1: "not X but Y" / "X, not Y"
    pattern_not_but = re.compile(
        r'not\s+([^,;:.!?]+?)\s*[,—–-]\s*(?:but\s+)?([^,;:.!?]+)',
        re.IGNORECASE
    )
    
    # Pattern 2: Parallel structures with dash/em-dash
    pattern_dash = re.compile(
        r'([^—–-]+?)\s*[—–-]\s*([^—–-]+)',
    )
    
    for sent in sentences:
        sent_lower = sent.lower()
        
        # Check for opposite word pairs
        for w1, w2 in opposites:
            if w1 in sent_lower and w2 in sent_lower:
                antitheses.append({
                    "type": "word_contrast",
                    "text": sent[:100],
                    "contrast": f"{w1} vs {w2}",
                })
                break
        
        # Check "not X but Y" pattern
        for match in pattern_not_but.finditer(sent):
            antitheses.append({
                "type": "not_but",
                "text": match.group(0)[:100],
                "parts": [match.group(1).strip(), match.group(2).strip()],
            })
        
        # Check for parallel structure with reversal
        # e.g., "X for Y" followed by "Y for X"
        words = word_tokenize(sent_lower)
        for i, word in enumerate(words):
            if word == 'for' and i > 0 and i < len(words) - 1:
                # Look for reversed pattern later
                for j in range(i + 2, len(words) - 1):
                    if words[j] == 'for':
                        # Check if pattern reverses: A for B ... B for A
                        if (words[i-1] == words[j+1] or words[i+1] == words[j-1]):
                            antitheses.append({
                                "type": "chiasmus",
                                "text": sent[:100],
                                "note": "reversed parallel structure",
                            })
                            break
    
    # Remove duplicates
    seen = set()
    unique_antitheses = []
    for a in antitheses:
        key = a.get("text", "")[:50]
        if key not in seen:
            seen.add(key)
            unique_antitheses.append(a)
    
    return {
        "antitheses": unique_antitheses[:10],
        "count": len(unique_antitheses),
    }


def detect_climax(transcript, sentiment_scores=None):
    """
    Detect climax - arrangement of ideas in order of increasing importance.
    
    Examples:
    - "I came, I saw, I conquered"
    - "...not only to survive, but to thrive, but to triumph!"
    
    Heuristics:
    1. Progressively longer phrases in a series
    2. Progressively more emphatic words
    3. Building intensity in sentiment
    """
    climaxes = []
    sentences = sent_tokenize(transcript)
    
    # Intensity words (ranked low to high)
    intensity_levels = {
        # Level 1 - mild
        'good': 1, 'nice': 1, 'fine': 1, 'okay': 1, 'some': 1,
        # Level 2 - moderate  
        'great': 2, 'important': 2, 'significant': 2, 'strong': 2,
        # Level 3 - strong
        'powerful': 3, 'tremendous': 3, 'incredible': 3, 'amazing': 3, 'essential': 3,
        # Level 4 - emphatic
        'critical': 4, 'vital': 4, 'ultimate': 4, 'supreme': 4, 'triumph': 4,
        # Level 5 - climactic
        'destiny': 5, 'forever': 5, 'eternal': 5, 'triumph': 5, 'victory': 5, 'glory': 5,
    }
    
    for sent in sentences:
        sent_lower = sent.lower()
        words = word_tokenize(sent_lower)
        
        # Look for comma-separated series
        parts = re.split(r'[,;]', sent)
        if len(parts) >= 3:
            # Check if parts are increasing in length
            lengths = [len(p.strip()) for p in parts if p.strip()]
            if len(lengths) >= 3:
                # Check if generally increasing
                increases = sum(1 for i in range(1, len(lengths)) if lengths[i] >= lengths[i-1])
                if increases >= len(lengths) - 2:  # Allow one dip
                    # Also check for intensity words
                    intensities = []
                    for part in parts:
                        part_words = word_tokenize(part.lower())
                        max_intensity = max(
                            (intensity_levels.get(w, 0) for w in part_words),
                            default=0
                        )
                        intensities.append(max_intensity)
                    
                    # If intensity is building or phrase length is building
                    intensity_building = all(intensities[i] >= intensities[i-1] 
                                            for i in range(1, len(intensities)) 
                                            if intensities[i] > 0)
                    
                    if intensity_building or (lengths[-1] > lengths[0] * 1.5):
                        climaxes.append({
                            "type": "building_series",
                            "text": sent[:150],
                            "parts_count": len(parts),
                        })
    
    # Also check for "not only... but also... but" patterns
    pattern_escalation = re.compile(
        r'not only\s+([^,]+),?\s*but\s+(?:also\s+)?([^,]+)(?:,?\s*but\s+([^,]+))?',
        re.IGNORECASE
    )
    
    for sent in sentences:
        for match in pattern_escalation.finditer(sent):
            climaxes.append({
                "type": "not_only_but",
                "text": match.group(0)[:100],
            })
    
    return {
        "climaxes": climaxes[:10],
        "count": len(climaxes),
    }


def analyze_all_devices(transcript, segments=None):
    """
    Comprehensive rhetorical device analysis.
    Returns all detected devices: anaphora, epistrophe, tricolon, 
    antithesis, rhetorical questions, and climax.
    """
    repetition = analyze_repetition(transcript, segments)
    tricolon = detect_tricolon(transcript)
    rhetorical_qs = detect_rhetorical_questions(transcript)
    antithesis = detect_antithesis(transcript)
    climax = detect_climax(transcript)
    
    # Calculate overall device score (0-100)
    device_count = (
        len(repetition.get("anaphora", [])) +
        len(repetition.get("epistrophe", [])) +
        tricolon.get("count", 0) +
        rhetorical_qs.get("rhetorical_count", 0) +
        antithesis.get("count", 0) +
        climax.get("count", 0)
    )
    
    # Score based on device density and variety
    device_types_used = sum([
        1 if repetition.get("anaphora") else 0,
        1 if repetition.get("epistrophe") else 0,
        1 if tricolon.get("count", 0) > 0 else 0,
        1 if rhetorical_qs.get("rhetorical_count", 0) > 0 else 0,
        1 if antithesis.get("count", 0) > 0 else 0,
        1 if climax.get("count", 0) > 0 else 0,
    ])
    
    # Score: count * 10 + variety bonus, capped at 100
    device_score = min(100, device_count * 10 + device_types_used * 10)
    
    return {
        "anaphora": repetition.get("anaphora", []),
        "epistrophe": repetition.get("epistrophe", []),
        "tricolon": tricolon.get("tricolons", []),
        "rhetorical_questions": rhetorical_qs.get("rhetorical_questions", []),
        "antithesis": antithesis.get("antitheses", []),
        "climax": climax.get("climaxes", []),
        "repetition_score": repetition.get("repetition_score", 0),
        "device_count": device_count,
        "device_types_used": device_types_used,
        "device_score": device_score,
        "summary": {
            "anaphora_count": len(repetition.get("anaphora", [])),
            "epistrophe_count": len(repetition.get("epistrophe", [])),
            "tricolon_count": tricolon.get("count", 0),
            "rhetorical_question_count": rhetorical_qs.get("rhetorical_count", 0),
            "antithesis_count": antithesis.get("count", 0),
            "climax_count": climax.get("count", 0),
        }
    }
