#!/usr/bin/env python3
"""
Generate pronunciation grade sample audio files using Edge TTS.

Creates reference audio samples demonstrating pronunciation quality levels A+ through F.
Uses different voices and post-processing (speed/pitch) to simulate quality variations.
"""

import os
import asyncio
import json
import subprocess
from pathlib import Path

import edge_tts

# Standard phrases for pronunciation demonstration
SAMPLE_PHRASES = {
    "clarity": [
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
        "Peter Piper picked a peck of pickled peppers.",
    ],
    "consonants": [
        "The thirty-three thieves thought they thrilled the throne.",
        "Red lorry, yellow lorry, red lorry, yellow lorry.",
        "Rubber baby buggy bumpers bounce brightly.",
    ],
    "vowels": [
        "How now brown cow, allow me to show you how.",
        "The rain in Spain stays mainly in the plain.",
        "Around the rugged rocks the ragged rascal ran.",
    ],
}

# Grade descriptions for metadata
GRADE_INFO = {
    "A+": {
        "score_range": "95-100",
        "description": "Exceptional clarity, perfect articulation, professional broadcast quality",
        "characteristics": ["Crystal clear consonants", "Perfect vowel formation", "Ideal pacing"],
    },
    "A": {
        "score_range": "90-94", 
        "description": "Excellent pronunciation, very clear and confident delivery",
        "characteristics": ["Clear articulation", "Consistent quality", "Natural rhythm"],
    },
    "B": {
        "score_range": "80-89",
        "description": "Good pronunciation with minor inconsistencies",
        "characteristics": ["Generally clear", "Occasional soft consonants", "Good overall"],
    },
    "C": {
        "score_range": "70-79",
        "description": "Average pronunciation, some unclear words",
        "characteristics": ["Some mumbling", "Inconsistent clarity", "Room for improvement"],
    },
    "D": {
        "score_range": "60-69",
        "description": "Below average, frequent unclear pronunciation",
        "characteristics": ["Frequent issues", "Rushed or slurred", "Needs practice"],
    },
    "F": {
        "score_range": "0-59",
        "description": "Poor pronunciation, difficult to understand",
        "characteristics": ["Very unclear", "Missing sounds", "Significant issues"],
    },
}

OUTPUT_DIR = Path("/home/melchior/speech3/static/pronunciation_samples")

# Use different voices to simulate quality variation
# High quality: professional news voices, Low quality: faster/less clear
VOICE_CONFIG = {
    "A+": {"voice": "en-US-GuyNeural", "rate": "-5%", "pitch": "+0Hz"},
    "A": {"voice": "en-US-JennyNeural", "rate": "+0%", "pitch": "+0Hz"},
    "B": {"voice": "en-US-AriaNeural", "rate": "+10%", "pitch": "-2Hz"},
    "C": {"voice": "en-US-ChristopherNeural", "rate": "+20%", "pitch": "-5Hz"},
    "D": {"voice": "en-US-EricNeural", "rate": "+35%", "pitch": "-8Hz"},
    "F": {"voice": "en-US-RogerNeural", "rate": "+50%", "pitch": "-10Hz"},
}


async def generate_sample(text: str, grade: str, category: str, index: int) -> str:
    """Generate a single audio sample using Edge TTS with grade-appropriate quality."""
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    filename = f"{grade}_{category}_{index}.mp3"
    output_path = OUTPUT_DIR / filename
    
    config = VOICE_CONFIG[grade]
    
    communicate = edge_tts.Communicate(
        text,
        config["voice"],
        rate=config["rate"],
        pitch=config["pitch"]
    )
    
    try:
        await communicate.save(str(output_path))
        print(f"  ✓ Generated {filename}")
        return str(output_path)
    except Exception as e:
        print(f"  ✗ Failed to generate {filename}: {e}")
        return None


async def generate_all_samples():
    """Generate samples for all grades and categories."""
    
    print("🎤 Generating pronunciation grade samples...\n")
    
    manifest = {"grades": {}, "categories": list(SAMPLE_PHRASES.keys())}
    
    for grade in GRADE_INFO.keys():
        print(f"\n📊 Grade {grade}: {GRADE_INFO[grade]['description']}")
        manifest["grades"][grade] = {
            "info": GRADE_INFO[grade],
            "samples": {}
        }
        
        for category, phrases in SAMPLE_PHRASES.items():
            manifest["grades"][grade]["samples"][category] = []
            
            for i, phrase in enumerate(phrases):
                path = await generate_sample(phrase, grade, category, i)
                if path:
                    manifest["grades"][grade]["samples"][category].append({
                        "file": os.path.basename(path),
                        "text": phrase,
                        "url": f"/static/pronunciation_samples/{os.path.basename(path)}"
                    })
    
    # Save manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n✅ Saved manifest to {manifest_path}")
    
    return manifest


def main():
    asyncio.run(generate_all_samples())


if __name__ == "__main__":
    main()
