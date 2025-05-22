import spacy
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from sentence_transformers import SentenceTransformer, util

nlp = spacy.load('en_core_web_sm')
model = SentenceTransformer('all-MiniLM-L6-v2')

transcript_low = "Uh… okay, so, um… there’s this picture, right? And, uh, it’s got a big sun, kinda orange, and, uh, the sky looks nice. There’s water, like a lake, I guess. And there’s a boat, just floating, not moving or anything. Trees are on the sides. Some birds up there, flying. Looks peaceful. It’s… um… nice, I think. Yeah, real nice. I like the colors. It’s, uh, a really good picture, I guess."

transcript_lowmid = "Okay, so in this picture, I see a sunset over a really calm lake. The sun is big and orange, and the sky has some pink and yellow. The water looks smooth, and there’s a small boat floating. Trees are on both sides, kinda dark. Some birds are flying above. Everything looks peaceful, like maybe it’s evening and getting dark. The colors are really pretty. It kinda makes me feel relaxed, like just sitting there."

transcript_mid = "This picture shows a beautiful sunset over a quiet lake. The sun is low, casting golden light across the water, which looks really smooth. A small boat floats there, empty, just drifting. Trees on either side make the scene feel enclosed, cozy. Birds fly in the distance, adding movement. Everything about it feels peaceful, like the end of a long day. It reminds me of times when I’ve just sat and watched the sunset."

transcript_highmid = "The image depicts a stunning sunset over a still lake, with golden hues reflecting across the water’s surface. A solitary boat drifts gently, giving a sense of quiet solitude. Surrounding trees frame the composition, their dark silhouettes contrasting against the glowing sky. Birds in flight add a subtle dynamic element. The scene conveys tranquility and introspection, almost like a moment suspended in time. It’s the kind of place where someone might sit and reflect."

transcript_high = "This image masterfully captures a moment of quiet beauty—a golden sunset dissolving into the lake’s mirrored surface. The unoccupied boat, adrift, suggests solitude, even longing. Trees frame the scene, their darkened forms grounding the composition. Birds in flight add energy, a fleeting contrast to the lake’s stillness. Light and shadow interact gracefully, creating depth and warmth. It’s a study in balance, stillness, and impermanence, evoking both peace and a touch of melancholy."


# Create array from all 
transcripts = [transcript_low, transcript_lowmid, transcript_mid, transcript_highmid, transcript_high]


# Analysis functions
def preposition_count(text):
    doc = nlp(text)
    return sum(1 for token in doc if token.pos_ == 'ADP')

def word_count(text):
    words = word_tokenize(text)
    return len(words)

def lexical_diversity(text):
    words = word_tokenize(text.lower())
    return len(set(words)) / len(words)

def syntactic_complexity(text):
    doc = nlp(text)
    sentences = list(doc.sents)
    lengths = [len(sent) for sent in sentences]
    return np.mean(lengths)

def speech_fluency(text):
    fillers = ['um', 'uh', 'er', 'ah', 'like', 'you know']
    words = word_tokenize(text.lower())
    filler_count = sum(words.count(filler) for filler in fillers)
    return filler_count

def connectedness_of_speech(text):
    sentences = sent_tokenize(text)
    embeddings = model.encode(sentences)
    similarities = [util.cos_sim(embeddings[i], embeddings[i+1]).item() for i in range(len(sentences)-1)]
    return np.mean(similarities)

def semantic_content_similarity(text, reference):
    embedding_text = model.encode(text)
    embedding_ref = model.encode(reference)
    return util.cos_sim(embedding_text, embedding_ref).item()

def circumlocution(text, target_terms):
    sentences = sent_tokenize(text)
    circumlocution_scores = []
    for term in target_terms:
        term_embedding = model.encode(term)
        for sent in sentences:
            sent_embedding = model.encode(sent)
            similarity = util.cos_sim(sent_embedding, term_embedding).item()
            if similarity > 0.5 and term not in sent:
                circumlocution_scores.append((sent, similarity))
    return circumlocution_scores


# For each of the transcripts, run the analyses
for i, transcript in enumerate(transcripts):

	# Print transcript name
	print(f"\nTranscript {i+1}")

	# Run analyses
	print("Preposition Count:", preposition_count(transcript))
	print("Word Count:", word_count(transcript))
	print("Lexical Diversity:", lexical_diversity(transcript))
	print("Syntactic Complexity (avg words per sentence):", syntactic_complexity(transcript))
	print("Speech Fluency (filler words count):", speech_fluency(transcript))
	print("Connectedness of Speech:", connectedness_of_speech(transcript))

	# Example reference text for semantic content
	reference_text = "A clear, direct description goes here."
	print("Semantic Content Similarity:", semantic_content_similarity(transcript, reference_text))

	# Example target terms for circumlocution detection
	target_terms = ["keys", "car", "house"]
	print("Circumlocution Instances:", circumlocution(transcript, target_terms))

