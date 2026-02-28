"""
Montreal Forced Aligner Integration
Provides phoneme-level alignment with timestamps and confidence scores.
"""

import subprocess
import json
import os
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
import re

# MFA conda environment path
MFA_CONDA = os.path.expanduser("~/miniconda3/bin/activate")
MFA_ENV = "mfa"

@dataclass
class PhonemeAlignment:
    phone: str
    start: float
    end: float
    
@dataclass
class WordAlignment:
    word: str
    start: float
    end: float
    phonemes: List[PhonemeAlignment]

@dataclass
class MFAResult:
    words: List[WordAlignment]
    duration: float

def run_mfa_align(audio_path: str, transcript: str) -> Optional[MFAResult]:
    """
    Run MFA alignment on an audio file with transcript.
    Returns word and phoneme-level alignments.
    """
    # Create temp directory structure for MFA
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_dir = Path(tmpdir) / "corpus"
        output_dir = Path(tmpdir) / "output"
        corpus_dir.mkdir()
        output_dir.mkdir()
        
        # Get base name from audio
        audio_name = Path(audio_path).stem
        
        # Copy audio file
        audio_ext = Path(audio_path).suffix
        shutil.copy(audio_path, corpus_dir / f"{audio_name}{audio_ext}")
        
        # Write transcript file (MFA expects .lab or .txt)
        transcript_file = corpus_dir / f"{audio_name}.lab"
        transcript_file.write_text(transcript.strip())
        
        # Run MFA alignment (TextGrid is default and most reliable format)
        cmd = f"""
        source {MFA_CONDA} {MFA_ENV} && \
        mfa align "{corpus_dir}" english_us_arpa english_us_arpa "{output_dir}" \
            --clean --single_speaker 2>&1
        """
        
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True,
            executable="/bin/bash"
        )
        
        if result.returncode != 0:
            print(f"MFA error: {result.stderr}")
            return None
        
        # Parse TextGrid output (default format)
        textgrid_output = output_dir / f"{audio_name}.TextGrid"
        if textgrid_output.exists():
            return parse_textgrid(textgrid_output)
        
        # Fallback to JSON if specified
        json_output = output_dir / f"{audio_name}.json"
        if json_output.exists():
            return parse_mfa_json(json_output)
            
        print(f"No output found in {output_dir}")
        return None

def parse_mfa_json(json_path: Path) -> MFAResult:
    """Parse MFA JSON output format."""
    with open(json_path) as f:
        data = json.load(f)
    
    words = []
    duration = 0
    
    # MFA JSON structure: tiers -> words, phones
    if "tiers" in data:
        word_tier = data["tiers"].get("words", {})
        phone_tier = data["tiers"].get("phones", {})
        
        word_entries = word_tier.get("entries", [])
        phone_entries = phone_tier.get("entries", [])
        
        for word_entry in word_entries:
            start, end, word_text = word_entry
            if not word_text or word_text == "":
                continue
                
            # Find phonemes within this word's time range
            word_phones = []
            for phone_entry in phone_entries:
                p_start, p_end, phone = phone_entry
                if p_start >= start and p_end <= end and phone:
                    word_phones.append(PhonemeAlignment(
                        phone=phone,
                        start=round(p_start, 3),
                        end=round(p_end, 3)
                    ))
            
            words.append(WordAlignment(
                word=word_text,
                start=round(start, 3),
                end=round(end, 3),
                phonemes=word_phones
            ))
            
            duration = max(duration, end)
    
    return MFAResult(words=words, duration=round(duration, 3))

def parse_textgrid(textgrid_path: Path) -> MFAResult:
    """Parse TextGrid format with word and phone tiers."""
    content = textgrid_path.read_text()
    
    words = []
    phones = []
    duration = 0
    
    interval_pattern = r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "([^"]*)"'
    
    # Extract phone intervals first
    phone_match = re.search(r'name = "phones".*?intervals: size = (\d+)(.*?)(?=item \[|$)', 
                           content, re.DOTALL)
    if phone_match:
        for match in re.finditer(interval_pattern, phone_match.group(2)):
            start, end, phone = float(match.group(1)), float(match.group(2)), match.group(3)
            if phone:
                phones.append(PhonemeAlignment(
                    phone=phone,
                    start=round(start, 3),
                    end=round(end, 3)
                ))
    
    # Extract word intervals
    word_match = re.search(r'name = "words".*?intervals: size = (\d+)(.*?)(?=item \[|$)', 
                          content, re.DOTALL)
    
    if word_match:
        for match in re.finditer(interval_pattern, word_match.group(2)):
            start, end, word_text = float(match.group(1)), float(match.group(2)), match.group(3)
            if word_text:
                # Find phones within this word's time range
                word_phones = [p for p in phones if p.start >= start and p.end <= end]
                
                words.append(WordAlignment(
                    word=word_text,
                    start=round(start, 3),
                    end=round(end, 3),
                    phonemes=word_phones
                ))
                duration = max(duration, end)
    
    return MFAResult(words=words, duration=round(duration, 3))

def align_audio(audio_path: str, transcript: Optional[str] = None) -> dict:
    """
    Main entry point for MFA alignment.
    If no transcript provided, uses Whisper to generate one.
    """
    if transcript is None:
        # Use faster-whisper for transcription
        from faster_whisper import WhisperModel
        from difflib import SequenceMatcher
        model = WhisperModel("large-v3", device="cuda", compute_type="float16")
        segments, _ = model.transcribe(audio_path, word_timestamps=True)
        
        # Filter hallucinated segments (duplicates, impossible WPM, low probability)
        filtered_texts = []
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            duration = seg.end - seg.start
            wpm = (len(text.split()) / duration * 60) if duration > 0 else 0
            if wpm > 300:  # Impossible speed
                continue
            if hasattr(seg, 'words') and seg.words:
                avg_prob = sum(w.probability for w in seg.words) / len(seg.words)
                if avg_prob < 0.1:  # Low confidence
                    continue
            # Check for duplicates
            is_dup = any(SequenceMatcher(None, text.lower(), prev.lower()).ratio() > 0.8 for prev in filtered_texts[-3:])
            if not is_dup:
                filtered_texts.append(text)
        
        transcript = " ".join(filtered_texts)
    
    result = run_mfa_align(audio_path, transcript)
    
    if result is None:
        return {"error": "MFA alignment failed"}
    
    return {
        "duration": result.duration,
        "word_count": len(result.words),
        "words": [
            {
                "word": w.word,
                "start": w.start,
                "end": w.end,
                "phonemes": [
                    {"phone": p.phone, "start": p.start, "end": p.end}
                    for p in w.phonemes
                ]
            }
            for w in result.words
        ]
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mfa_alignment.py <audio_file> [transcript]")
        sys.exit(1)
    
    audio = sys.argv[1]
    transcript = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = align_audio(audio, transcript)
    print(json.dumps(result, indent=2))
