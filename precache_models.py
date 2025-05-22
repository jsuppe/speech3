#!/usr/bin/env python3
"""
Script to precache models for faster container startup.
This downloads and initializes the models during Docker image build.
"""
import os
import sys
import time
print("Starting model precaching...")

# Set cache environment variables
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/app/models"
os.environ["TRANSFORMERS_CACHE"] = "/app/models"
os.environ["HF_HOME"] = "/app/models"

# Import libraries only after setting cache directories
try:
    import spacy
    from sentence_transformers import SentenceTransformer
    import nltk
    print("Successfully imported libraries")
except ImportError as e:
    print(f"Error importing libraries: {e}")
    sys.exit(1)

start_time = time.time()

# Load and cache the models
try:
    # Initialize spaCy model
    print("Loading spaCy model...")
    nlp = spacy.load('en_core_web_sm')
    print(f"✅ SpaCy model loaded. Type: {type(nlp)}")
    
    # Download NLTK data
    print("Downloading NLTK data...")
    nltk.download('punkt', download_dir='/app/models/nltk_data')
    os.environ['NLTK_DATA'] = '/app/models/nltk_data'
    print("✅ NLTK data downloaded")
    
    # Initialize the sentence transformer model
    print("Loading sentence transformer model...")
    model_name = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)
    print(f"✅ Sentence transformer model loaded. Type: {type(model)}")
    
    # Test the model with a sample input
    print("Testing model with sample input...")
    embedding = model.encode("This is a test sentence.")
    print(f"✅ Model test successful. Embedding shape: {embedding.shape}")
    
    end_time = time.time()
    print(f"✅ All models precached successfully in {end_time - start_time:.2f} seconds")
    
except Exception as e:
    print(f"❌ Error precaching models: {e}")
    sys.exit(1)
