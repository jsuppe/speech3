"""
Speaker fingerprinting module for SpeechScore.

Uses Resemblyzer to extract voice embeddings (256-dim vectors) that can be used
to identify speakers across recordings. The embeddings capture the unique
characteristics of a speaker's voice.

Usage:
    from speaker_fingerprint import extract_embedding, identify_speaker

    # Extract embedding from audio
    embedding = extract_embedding("/path/to/audio.wav")

    # Find matching speaker from stored embeddings
    match = identify_speaker(db, user_id, embedding)
    if match:
        print(f"Detected speaker: {match['speaker_name']} ({match['similarity']:.0%} confidence)")
"""

import os
import logging
import tempfile
import subprocess
import numpy as np
from typing import Optional, Tuple

logger = logging.getLogger("speechscore.fingerprint")

# Lazy-loaded encoder (same as diarization.py)
_encoder = None


def _get_encoder():
    """Lazy load the Resemblyzer voice encoder."""
    global _encoder
    if _encoder is None:
        from resemblyzer import VoiceEncoder
        _encoder = VoiceEncoder()
        logger.info("Resemblyzer VoiceEncoder loaded for fingerprinting")
    return _encoder


def _convert_to_wav_16k(audio_path: str) -> str:
    """Convert audio to 16kHz mono WAV for resemblyzer."""
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", wav_path],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {result.stderr[:200]}")
        return wav_path
    except Exception:
        if os.path.exists(wav_path):
            os.remove(wav_path)
        raise


def extract_embedding(audio_path: str, min_duration: float = 1.0) -> Optional[bytes]:
    """
    Extract a voice embedding from an audio file.
    
    Args:
        audio_path: Path to audio file (any format ffmpeg supports)
        min_duration: Minimum audio duration in seconds (default 1.0)
    
    Returns:
        256-dim float32 embedding as bytes, or None if extraction failed
    """
    encoder = _get_encoder()
    
    # Convert to WAV if needed
    wav_path = None
    needs_cleanup = False
    
    try:
        if not audio_path.lower().endswith('.wav'):
            wav_path = _convert_to_wav_16k(audio_path)
            needs_cleanup = True
        else:
            # Still need to ensure 16kHz mono
            wav_path = _convert_to_wav_16k(audio_path)
            needs_cleanup = True
        
        from resemblyzer import preprocess_wav
        wav = preprocess_wav(wav_path)
        
        # Check minimum duration
        duration = len(wav) / 16000
        if duration < min_duration:
            logger.warning(f"Audio too short for fingerprinting: {duration:.1f}s < {min_duration}s")
            return None
        
        # Extract embedding from the entire utterance
        embedding = encoder.embed_utterance(wav)
        
        # Normalize (should already be normalized, but ensure)
        embedding = embedding / np.linalg.norm(embedding)
        
        logger.debug(f"Extracted voice embedding: shape={embedding.shape}, norm={np.linalg.norm(embedding):.3f}")
        
        return embedding.astype(np.float32).tobytes()
    
    except Exception as e:
        logger.error(f"Failed to extract voice embedding: {e}")
        return None
    
    finally:
        if needs_cleanup and wav_path and os.path.exists(wav_path):
            os.remove(wav_path)


def identify_speaker(
    db,
    user_id: int,
    embedding: bytes,
    threshold: float = 0.80,
) -> Optional[dict]:
    """
    Identify a speaker by comparing their voice embedding to stored profiles.
    
    Args:
        db: SpeechDB instance
        user_id: User ID to search within
        embedding: Voice embedding bytes (from extract_embedding)
        threshold: Minimum similarity for a match (0.80 = 80% confidence)
    
    Returns:
        Dict with speaker_name and similarity, or None if no match
    """
    if not embedding:
        return None
    
    return db.find_matching_speaker(user_id, embedding, threshold=threshold)


def register_speaker(
    db,
    user_id: int,
    speaker_name: str,
    audio_path: str,
    speech_id: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    Register a speaker's voice from an audio file.
    
    If the speaker already exists, their embedding will be updated (averaged)
    for better accuracy.
    
    Args:
        db: SpeechDB instance
        user_id: User ID who owns this speaker profile
        speaker_name: Display name for the speaker
        audio_path: Path to audio file
        speech_id: Optional reference to the speech record
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    embedding = extract_embedding(audio_path)
    if not embedding:
        return False, "Could not extract voice embedding from audio"
    
    try:
        embed_id = db.store_speaker_embedding(
            user_id=user_id,
            speaker_name=speaker_name,
            embedding=embedding,
            speech_id=speech_id,
        )
        return True, f"Speaker '{speaker_name}' registered (embedding_id={embed_id})"
    except Exception as e:
        logger.error(f"Failed to store speaker embedding: {e}")
        return False, f"Failed to store embedding: {str(e)}"


def process_audio_for_speaker(
    db,
    user_id: int,
    audio_path: str,
    provided_speaker: Optional[str] = None,
    speech_id: Optional[int] = None,
    auto_register: bool = True,
    match_threshold: float = 0.80,
) -> dict:
    """
    Process audio to identify or register speaker.
    
    This is the main entry point for speaker fingerprinting. It:
    1. Extracts the voice embedding from the audio
    2. If speaker is provided: registers/updates the speaker profile
    3. If speaker not provided: tries to match against known speakers
    
    Args:
        db: SpeechDB instance
        user_id: User ID
        audio_path: Path to audio file
        provided_speaker: Speaker name if already known
        speech_id: Speech record ID (for linking)
        auto_register: If speaker is provided, automatically register their voice
        match_threshold: Minimum similarity for auto-detection (0.80 = 80%)
    
    Returns:
        Dict with:
            - embedding_extracted: bool
            - speaker_identified: str or None (detected speaker name)
            - speaker_registered: bool (if we stored the embedding)
            - similarity: float or None (match confidence)
            - is_new_speaker: bool (true if this is a new voice profile)
    """
    result = {
        "embedding_extracted": False,
        "speaker_identified": None,
        "speaker_registered": False,
        "similarity": None,
        "is_new_speaker": False,
    }
    
    # Extract embedding
    embedding = extract_embedding(audio_path)
    if not embedding:
        logger.warning("Could not extract voice embedding")
        return result
    
    result["embedding_extracted"] = True
    
    if provided_speaker:
        # Speaker is known - register their voice
        if auto_register:
            # Check if this is a new speaker
            existing = db.find_matching_speaker(user_id, embedding, threshold=0.5)
            if not existing or existing["speaker_name"] != provided_speaker:
                result["is_new_speaker"] = True
            
            db.store_speaker_embedding(
                user_id=user_id,
                speaker_name=provided_speaker,
                embedding=embedding,
                speech_id=speech_id,
            )
            result["speaker_registered"] = True
        
        result["speaker_identified"] = provided_speaker
        result["similarity"] = 1.0  # Provided, not matched
    else:
        # Try to identify speaker
        match = db.find_matching_speaker(user_id, embedding, threshold=match_threshold)
        if match:
            result["speaker_identified"] = match["speaker_name"]
            result["similarity"] = match["similarity"]
            logger.info(
                f"Voice match: {match['speaker_name']} "
                f"(similarity={match['similarity']:.1%})"
            )
    
    return result


# CLI for testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python speaker_fingerprint.py extract <audio_file>")
        print("  python speaker_fingerprint.py compare <audio1> <audio2>")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO)
    
    cmd = sys.argv[1]
    
    if cmd == "extract" and len(sys.argv) >= 3:
        audio_path = sys.argv[2]
        print(f"Extracting embedding from: {audio_path}")
        embedding = extract_embedding(audio_path)
        if embedding:
            arr = np.frombuffer(embedding, dtype=np.float32)
            print(f"  Shape: {arr.shape}")
            print(f"  Norm: {np.linalg.norm(arr):.4f}")
            print(f"  First 10 dims: {arr[:10]}")
        else:
            print("  Failed to extract embedding")
    
    elif cmd == "compare" and len(sys.argv) >= 4:
        audio1, audio2 = sys.argv[2], sys.argv[3]
        print(f"Comparing: {audio1} vs {audio2}")
        
        emb1 = extract_embedding(audio1)
        emb2 = extract_embedding(audio2)
        
        if emb1 and emb2:
            arr1 = np.frombuffer(emb1, dtype=np.float32)
            arr2 = np.frombuffer(emb2, dtype=np.float32)
            similarity = float(np.dot(arr1, arr2))
            print(f"  Cosine similarity: {similarity:.4f} ({similarity*100:.1f}%)")
            
            if similarity >= 0.85:
                print("  → High confidence: SAME SPEAKER")
            elif similarity >= 0.75:
                print("  → Medium confidence: Probably same speaker")
            elif similarity >= 0.65:
                print("  → Low confidence: Possibly same speaker")
            else:
                print("  → Different speakers")
        else:
            print("  Failed to extract one or both embeddings")
    
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
