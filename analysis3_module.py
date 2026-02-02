"""speech3 analysis as an importable module."""
import os, sys
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import spacy
nlp = spacy.load('en_core_web_sm')

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')


def preposition_count(text):
    doc = nlp(text)
    return sum(1 for token in doc if token.pos_ == 'ADP')

def word_count(text):
    return len(word_tokenize(text))

def lexical_diversity(text):
    words = word_tokenize(text.lower())
    return round(len(set(words)) / len(words), 4)

def syntactic_complexity(text):
    doc = nlp(text)
    sentences = list(doc.sents)
    lengths = [len(sent) for sent in sentences]
    return round(float(np.mean(lengths)), 2)

def speech_fluency(text):
    fillers = ['um', 'uh', 'er', 'ah', 'like', 'you know']
    words = word_tokenize(text.lower())
    return sum(words.count(f) for f in fillers)

def connectedness_of_speech(text):
    sentences = sent_tokenize(text)
    if len(sentences) <= 1:
        return 0.0
    embeddings = model.encode(sentences)
    sims = [util.cos_sim(embeddings[i], embeddings[i+1]).item() for i in range(len(sentences)-1)]
    return round(float(np.mean(sims)), 4)

def mean_length_of_utterance(text):
    sentences = sent_tokenize(text)
    lengths = [len(word_tokenize(sent)) for sent in sentences]
    return round(float(np.mean(lengths)), 2)

def pronoun_usage(text):
    doc = nlp(text)
    return sum(1 for token in doc if token.pos_ == 'PRON')

def negation_usage(text):
    negations = ['no', 'not', 'never']
    words = word_tokenize(text.lower())
    return sum(words.count(neg) for neg in negations)

def extract_key_terms(text):
    doc = nlp(text)
    exclude = ['moment', 'scene', 'ambiance']
    return list(set(token.text.lower() for token in doc if token.pos_ == 'NOUN' and token.text.lower() not in exclude))

def circumlocution(text, target_terms):
    sentences = sent_tokenize(text)
    results = []
    for term in target_terms:
        term_emb = model.encode(term)
        for sent in sentences:
            sent_emb = model.encode(sent)
            sim = util.cos_sim(sent_emb, term_emb).item()
            if sim > 0.3 and term not in sent.lower():
                results.append({"sentence": sent, "similarity": round(sim, 4), "term": term})
    return results


def analyze_transcript(text):
    key_terms = extract_key_terms(text)
    circ = circumlocution(text, key_terms)
    return {
        "metrics": {
            "word_count": word_count(text),
            "lexical_diversity": lexical_diversity(text),
            "syntactic_complexity": syntactic_complexity(text),
            "speech_fluency_fillers": speech_fluency(text),
            "connectedness": connectedness_of_speech(text),
            "mean_utterance_length": mean_length_of_utterance(text),
            "prepositions": preposition_count(text),
            "pronouns": pronoun_usage(text),
            "negations": negation_usage(text),
            "circumlocution_score": len(circ),
        },
        "key_terms": key_terms,
    }
