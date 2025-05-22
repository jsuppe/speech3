import sys
import os
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
import subprocess
import os.path

# Check for spaCy model before importing
try:
    import spacy
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        print("Error: SpaCy model 'en_core_web_sm' is not installed.")
        print("Please run: python setup.py")
        print("Or install it directly with: python -m spacy download en_core_web_sm")
        sys.exit(1)
except ImportError:
    print("Error: SpaCy is not installed.")
    print("Please run: python setup.py")
    sys.exit(1)

# Set environment variables for model caching
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/app/models"
os.environ["TRANSFORMERS_CACHE"] = "/app/models"
os.environ["HF_HOME"] = "/app/models"
os.environ['NLTK_DATA'] = '/app/models/nltk_data'

# Check for sentence_transformers
try:
    from sentence_transformers import SentenceTransformer, util
    print("Loading pre-cached model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded successfully!")
except ImportError:
    print("Error: sentence_transformers is not installed.")
    print("Please run: python setup.py")
    sys.exit(1)

# Sample transcripts
transcript_low = "Uh… okay, so, um… there’s this picture, right? And, uh, it’s got a big sun, kinda orange, and, uh, the sky looks nice. There’s water, like a lake, I guess. And there’s a boat, just floating, not moving or anything. Trees are on the sides. Some birds up there, flying. Looks peaceful. It’s… um… nice, I think. Yeah, real nice. I like the colors. It’s, uh, a really good picture, I guess."

transcript_lowmid = "Okay, so in this picture, I see a sunset over a really calm lake. The sun is big and orange, and the sky has some pink and yellow. The water looks smooth, and there’s a small boat floating. Trees are on both sides, kinda dark. Some birds are flying above. Everything looks peaceful, like maybe it’s evening and getting dark. The colors are really pretty. It kinda makes me feel relaxed, like just sitting there."

transcript_mid = "This picture shows a beautiful sunset over a quiet lake. The sun is low, casting golden light across the water, which looks really smooth. A small boat floats there, empty, just drifting. Trees on either side make the scene feel enclosed, cozy. Birds fly in the distance, adding movement. Everything about it feels peaceful, like the end of a long day. It reminds me of times when I’ve just sat and watched the sunset."

transcript_highmid = "The image depicts a stunning sunset over a still lake, with golden hues reflecting across the water’s surface. A solitary boat drifts gently, giving a sense of quiet solitude. Surrounding trees frame the composition, their dark silhouettes contrasting against the glowing sky. Birds in flight add a subtle dynamic element. The scene conveys tranquility and introspection, almost like a moment suspended in time. It’s the kind of place where someone might sit and reflect."

transcript_high = "This image masterfully captures a moment of quiet beauty—a golden sunset dissolving into the lake’s mirrored surface. The unoccupied boat, adrift, suggests solitude, even longing. Trees frame the scene, their darkened forms grounding the composition. Birds in flight add energy, a fleeting contrast to the lake’s stillness. Light and shadow interact gracefully, creating depth and warmth. It’s a study in balance, stillness, and impermanence, evoking both peace and a touch of melancholy."

# Create array from all transcripts
default_transcripts = [transcript_low, transcript_lowmid, transcript_mid, transcript_highmid, transcript_high]

# Analysis functions
def preposition_count(text):
    """Counts the number of prepositions in a given text."""
    doc = nlp(text)
    return sum(1 for token in doc if token.pos_ == 'ADP')

def word_count(text):
    """Counts the number of words in a given text."""
    words = word_tokenize(text)
    return len(words)

def lexical_diversity(text):
    """Calculates the lexical diversity of a text."""
    words = word_tokenize(text.lower())
    return len(set(words)) / len(words)

def syntactic_complexity(text):
    """Measures the syntactic complexity of a text by calculating the average sentence length."""
    doc = nlp(text)
    sentences = list(doc.sents)
    lengths = [len(sent) for sent in sentences]
    return np.mean(lengths)

def speech_fluency(text):
    """Evaluates speech fluency by counting the number of filler words in the text."""
    fillers = ['um', 'uh', 'er', 'ah', 'like', 'you know']
    words = word_tokenize(text.lower())
    filler_count = sum(words.count(filler) for filler in fillers)
    return filler_count

def connectedness_of_speech(text):
    """Measures the connectedness of speech by calculating the average cosine similarity between consecutive sentences."""
    sentences = sent_tokenize(text)
    embeddings = model.encode(sentences)
    similarities = [util.cos_sim(embeddings[i], embeddings[i+1]).item() for i in range(len(sentences)-1)]
    return np.mean(similarities)

def semantic_content_similarity(text, reference):
    """Compares the semantic similarity between the input text and a reference text using cosine similarity of their embeddings."""
    embedding_text = model.encode(text)
    embedding_ref = model.encode(reference)
    return util.cos_sim(embedding_text, embedding_ref).item()

def circumlocution(text, target_terms):
    """Identifies circumlocution (indirect or verbose expressions) by checking if sentences are semantically similar to target terms but do not explicitly contain them."""
    sentences = sent_tokenize(text)
    circumlocution_scores = []
    for term in target_terms:
        term_embedding = model.encode(term)
        for sent in sentences:
            sent_embedding = model.encode(sent)
            similarity = util.cos_sim(sent_embedding, term_embedding).item()
            # Lower the threshold to 0.3
            if similarity > 0.3 and term not in sent.lower():
                circumlocution_scores.append((sent, similarity))
    return circumlocution_scores

def extract_key_terms(text):
    """Extracts key terms (nouns) from the text, excluding generic terms like 'moment' and 'scene'."""
    doc = nlp(text)
    exclude_terms = ['moment', 'scene', 'ambiance']  # Terms to exclude
    return list(set(token.text.lower() for token in doc if token.pos_ == 'NOUN' and token.text.lower() not in exclude_terms))

# Additional indicators
def mean_length_of_utterance(text):
    """Calculates the mean length of utterance in the text."""
    sentences = sent_tokenize(text)
    lengths = [len(word_tokenize(sent)) for sent in sentences]
    return np.mean(lengths)

def pronoun_usage(text):
    """Counts the number of pronouns used in the text."""
    doc = nlp(text)
    return sum(1 for token in doc if token.pos_ == 'PRON')

def negation_usage(text):
    """Counts the number of negations used in the text."""
    negations = ['no', 'not', 'never']
    words = word_tokenize(text.lower())
    return sum(words.count(neg) for neg in negations)

# Main execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run speech analysis.")
    parser.add_argument("transcript", help="Transcript text or path to transcript file")
    parser.add_argument("--output", default=os.environ.get('RESULTS_FILE_PATH', 'output/results.txt'),
                       help="Path to the output results file")
    args = parser.parse_args()

    # Check if the argument is a file path and read from it, otherwise treat as direct text
    transcript = args.transcript
    if os.path.isfile(transcript):
        try:
            with open(transcript, "r") as f:
                transcript = f.read()
            print(f"Successfully read transcript from file: {args.transcript}")
        except Exception as e:
            print(f"Error reading transcript file: {e}")
            sys.exit(1)
    else:
        print("Processing transcript provided directly as argument")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run the analysis
    results_file_path = args.output
    with open(results_file_path, "w") as f:
        print(f"Processing Transcript: {transcript[:50]}...")  # Debug: Print the first 50 characters of the transcript
        f.write(f"\nTranscript\n")
        f.write(f"Preposition Count: {preposition_count(transcript)}\n")
        f.write(f"Word Count: {word_count(transcript)}\n")
        f.write(f"Lexical Diversity: {lexical_diversity(transcript)}\n")
        f.write(f"Syntactic Complexity: {syntactic_complexity(transcript)}\n")
        f.write(f"Speech Fluency: {speech_fluency(transcript)}\n")
        f.write(f"Connectedness of Speech: {connectedness_of_speech(transcript)}\n")
        f.write(f"Mean Length of Utterance: {mean_length_of_utterance(transcript)}\n")
        f.write(f"Pronoun Usage: {pronoun_usage(transcript)}\n")
        f.write(f"Negation Usage: {negation_usage(transcript)}\n")
        
        # Extract key terms and calculate circumlocution score
        key_terms = extract_key_terms(transcript)
        f.write(f"Extracted Key Terms: {key_terms}\n")
        circumlocution_results = circumlocution(transcript, key_terms)
        circumlocution_score = len(circumlocution_results)
        f.write(f"Circumlocution Score: {circumlocution_score}\n")
        if circumlocution_results:
            f.write(f"Circumlocution Details: {circumlocution_results}\n")
        print(f"Finished processing Transcript. Results saved to {results_file_path}")



