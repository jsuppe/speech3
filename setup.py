"""
Setup script to install dependencies for speech analysis.
"""
import subprocess
import sys
import importlib
import os

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    print("Setting up speech analysis environment...")
    
    # Check for required packages
    required_packages = [
        "spacy",
        "numpy",
        "nltk",
        "sentence_transformers"
    ]
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} is already installed")
        except ImportError:
            print(f"📦 Installing {package}...")
            install_package(package)

    # Install SpaCy language model
    print("📦 Installing SpaCy language model...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    
    # Download NLTK resources
    print("📦 Downloading NLTK resources...")
    import nltk
    nltk.download('punkt')
    
    print("\n✅ Setup complete! You can now run the analysis scripts.")

if __name__ == "__main__":
    main()
