"""
Speaker diarization module using resemblyzer for speaker embeddings
and agglomerative clustering to identify distinct speakers.
"""

import os
import logging
import tempfile
import subprocess
import numpy as np
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger("speechscore.diarization")

# Lazy-loaded encoder
_encoder = None


def _get_encoder():
    """Lazy load the voice encoder."""
    global _encoder
    if _encoder is None:
        from resemblyzer import VoiceEncoder
        _encoder = VoiceEncoder()
        logger.info("Resemblyzer VoiceEncoder loaded")
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


def run_diarization(
    audio_path: str,
    segments: List[Dict],
    num_speakers: Optional[int] = None,
    min_segment_duration: float = 0.5,
    known_speaker: Optional[str] = None,
) -> List[Dict]:
    """
    Run speaker diarization on audio and assign speaker labels to segments.
    
    Args:
        audio_path: Path to audio file
        segments: List of transcript segments with 'start', 'end', 'text'
        num_speakers: Fixed number of speakers (None = auto-detect)
        min_segment_duration: Minimum segment duration to process (seconds)
        known_speaker: Name of primary speaker from metadata (used if single speaker detected)
    
    Returns:
        Updated segments list with 'speaker' field added
    """
    if not segments:
        return segments
    
    encoder = _get_encoder()
    
    # Convert to WAV if needed
    wav_path = None
    needs_cleanup = False
    
    if not audio_path.lower().endswith('.wav'):
        wav_path = _convert_to_wav_16k(audio_path)
        needs_cleanup = True
    else:
        wav_path = audio_path
    
    try:
        from resemblyzer import preprocess_wav
        wav = preprocess_wav(wav_path)
        
        # Extract embeddings for each segment
        embeddings = []
        valid_indices = []
        
        for i, seg in enumerate(segments):
            start_sample = int(seg['start'] * 16000)
            end_sample = int(seg['end'] * 16000)
            
            # Bounds check
            if end_sample > len(wav):
                end_sample = len(wav)
            
            duration = (end_sample - start_sample) / 16000
            if duration < min_segment_duration:
                continue
            
            segment_wav = wav[start_sample:end_sample]
            
            try:
                embed = encoder.embed_utterance(segment_wav)
                embeddings.append(embed)
                valid_indices.append(i)
            except Exception as e:
                logger.debug(f"Skipping segment {i} at {seg['start']:.1f}s: {e}")
        
        if len(embeddings) < 2:
            # Not enough segments for clustering
            logger.info("Not enough segments for diarization")
            return segments
        
        embeddings_array = np.array(embeddings)
        
        # Cluster speakers
        from sklearn.cluster import AgglomerativeClustering
        
        if num_speakers:
            clustering = AgglomerativeClustering(
                n_clusters=num_speakers,
                linkage='ward'
            )
        else:
            # Auto-detect number of speakers
            # Lower threshold (0.5) to better separate similar-sounding speakers
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.5,
                linkage='average'
            )
        
        labels = clustering.fit_predict(embeddings_array)
        n_speakers = len(set(labels))
        
        logger.info(f"Diarization: detected {n_speakers} speakers in {len(segments)} segments")
        
        # Assign speaker labels to segments
        # Use ordinal names for unknown speakers, or known_speaker if single speaker
        ordinals = [
            "First Speaker", "Second Speaker", "Third Speaker", "Fourth Speaker",
            "Fifth Speaker", "Sixth Speaker", "Seventh Speaker", "Eighth Speaker",
            "Ninth Speaker", "Tenth Speaker"
        ]
        
        # If single speaker detected and we know their name, use it
        use_known_speaker = n_speakers == 1 and known_speaker
        
        speaker_labels = {}
        for idx, label in zip(valid_indices, labels):
            if use_known_speaker:
                speaker_name = known_speaker
            else:
                speaker_name = ordinals[label] if label < len(ordinals) else f"Speaker {label + 1}"
            speaker_labels[idx] = speaker_name
        
        # Update segments with speaker info
        updated_segments = []
        for i, seg in enumerate(segments):
            new_seg = seg.copy()
            if i in speaker_labels:
                new_seg['speaker'] = speaker_labels[i]
            updated_segments.append(new_seg)
        
        return updated_segments
    
    finally:
        if needs_cleanup and wav_path and os.path.exists(wav_path):
            os.remove(wav_path)


def get_speaker_stats(segments: List[Dict]) -> Dict:
    """Get statistics about speakers in the transcript."""
    speakers = {}
    
    for seg in segments:
        speaker = seg.get('speaker', 'UNKNOWN')
        if speaker not in speakers:
            speakers[speaker] = {
                'segment_count': 0,
                'total_duration': 0.0,
                'word_count': 0,
            }
        
        speakers[speaker]['segment_count'] += 1
        speakers[speaker]['total_duration'] += seg['end'] - seg['start']
        speakers[speaker]['word_count'] += len(seg.get('text', '').split())
    
    return {
        'num_speakers': len(speakers),
        'speakers': speakers,
    }


def register_diarized_speakers(
    db,
    user_id: int,
    audio_path: str,
    segments: List[Dict],
    speech_id: Optional[int] = None,
    min_duration_per_speaker: float = 3.0,
) -> Dict:
    """
    Extract voice embeddings for each diarized speaker and register as fingerprints.
    
    This allows future recordings to auto-identify returning speakers.
    
    Args:
        db: SpeechDB instance
        user_id: User ID
        audio_path: Path to audio file
        segments: Diarized segments with 'speaker', 'start', 'end'
        speech_id: Speech record ID (for linking)
        min_duration_per_speaker: Minimum total duration to create profile (seconds)
    
    Returns:
        Dict with registration results for each speaker
    """
    from speaker_fingerprint import extract_embedding
    
    encoder = _get_encoder()
    
    # Group segments by speaker
    speaker_segments = {}
    for seg in segments:
        speaker = seg.get('speaker')
        if not speaker or speaker == 'UNKNOWN':
            continue
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
        speaker_segments[speaker].append(seg)
    
    if not speaker_segments:
        logger.info("No labeled speakers to register")
        return {"registered": [], "skipped": []}
    
    # Convert audio to WAV
    wav_path = None
    needs_cleanup = False
    
    if not audio_path.lower().endswith('.wav'):
        wav_path = _convert_to_wav_16k(audio_path)
        needs_cleanup = True
    else:
        wav_path = audio_path
    
    try:
        from resemblyzer import preprocess_wav
        wav = preprocess_wav(wav_path)
        
        results = {"registered": [], "skipped": [], "matched": []}
        
        for speaker_name, segs in speaker_segments.items():
            # Calculate total duration
            total_duration = sum(s['end'] - s['start'] for s in segs)
            
            if total_duration < min_duration_per_speaker:
                results["skipped"].append({
                    "speaker": speaker_name,
                    "reason": f"Insufficient duration ({total_duration:.1f}s < {min_duration_per_speaker}s)"
                })
                continue
            
            # Concatenate all segments for this speaker
            speaker_samples = []
            for seg in segs:
                start_sample = int(seg['start'] * 16000)
                end_sample = int(seg['end'] * 16000)
                if end_sample > len(wav):
                    end_sample = len(wav)
                speaker_samples.extend(wav[start_sample:end_sample])
            
            if len(speaker_samples) < 16000:  # Less than 1 second
                results["skipped"].append({
                    "speaker": speaker_name,
                    "reason": "Audio too short after extraction"
                })
                continue
            
            # Extract embedding
            try:
                speaker_wav = np.array(speaker_samples)
                embedding = encoder.embed_utterance(speaker_wav)
                embedding_bytes = embedding.astype(np.float32).tobytes()
                
                # Check if this voice matches an existing profile
                existing_profiles = db.get_speaker_embeddings(user_id)
                best_match = None
                best_similarity = 0.0
                
                for profile in existing_profiles:
                    stored_embedding = np.frombuffer(profile['embedding'], dtype=np.float32)
                    similarity = float(np.dot(embedding, stored_embedding))
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = profile
                
                # If high match (>85%), link to existing profile instead of creating new
                if best_match and best_similarity > 0.85:
                    # Update existing profile with averaged embedding
                    db.register_speaker_embedding(
                        user_id=user_id,
                        speaker_name=best_match['speaker_name'],
                        embedding=embedding_bytes,
                        speech_id=speech_id,
                    )
                    results["matched"].append({
                        "detected_as": speaker_name,
                        "matched_to": best_match['speaker_name'],
                        "similarity": round(best_similarity, 3),
                        "duration": round(total_duration, 1),
                    })
                    logger.info(f"Matched '{speaker_name}' to existing profile '{best_match['speaker_name']}' ({best_similarity:.0%})")
                else:
                    # Register as new speaker
                    db.register_speaker_embedding(
                        user_id=user_id,
                        speaker_name=speaker_name,
                        embedding=embedding_bytes,
                        speech_id=speech_id,
                    )
                    results["registered"].append({
                        "speaker": speaker_name,
                        "duration": round(total_duration, 1),
                        "is_new": True,
                    })
                    logger.info(f"Registered new speaker profile: '{speaker_name}' ({total_duration:.1f}s)")
                    
            except Exception as e:
                logger.warning(f"Failed to extract embedding for '{speaker_name}': {e}")
                results["skipped"].append({
                    "speaker": speaker_name,
                    "reason": str(e)
                })
        
        return results
        
    finally:
        if needs_cleanup and wav_path and os.path.exists(wav_path):
            os.remove(wav_path)


def match_speakers_to_profiles(
    db,
    user_id: int,
    audio_path: str,
    segments: List[Dict],
    match_threshold: float = 0.80,
) -> Tuple[List[Dict], Dict]:
    """
    Match diarized speakers against stored voice profiles and relabel.
    
    Args:
        db: SpeechDB instance
        user_id: User ID
        audio_path: Path to audio file
        segments: Diarized segments with 'speaker' labels
        match_threshold: Minimum similarity to accept match (0.80 = 80%)
    
    Returns:
        Tuple of (updated segments, match results dict)
    """
    encoder = _get_encoder()
    
    # Get stored profiles for this user
    stored_profiles = db.get_speaker_embeddings(user_id)
    if not stored_profiles:
        return segments, {"profiles_checked": 0, "matches": []}
    
    # Group segments by speaker
    speaker_segments = {}
    for i, seg in enumerate(segments):
        speaker = seg.get('speaker')
        if not speaker or speaker == 'UNKNOWN':
            continue
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
        speaker_segments[speaker].append((i, seg))
    
    if not speaker_segments:
        return segments, {"profiles_checked": len(stored_profiles), "matches": []}
    
    # Convert audio to WAV
    wav_path = None
    needs_cleanup = False
    
    if not audio_path.lower().endswith('.wav'):
        wav_path = _convert_to_wav_16k(audio_path)
        needs_cleanup = True
    else:
        wav_path = audio_path
    
    try:
        from resemblyzer import preprocess_wav
        wav = preprocess_wav(wav_path)
        
        matches = []
        speaker_remap = {}  # old_name -> new_name
        
        for speaker_name, indexed_segs in speaker_segments.items():
            # Extract samples for this speaker
            speaker_samples = []
            for _, seg in indexed_segs:
                start_sample = int(seg['start'] * 16000)
                end_sample = int(seg['end'] * 16000)
                if end_sample > len(wav):
                    end_sample = len(wav)
                speaker_samples.extend(wav[start_sample:end_sample])
            
            if len(speaker_samples) < 8000:  # Less than 0.5 seconds
                continue
            
            # Extract embedding
            try:
                speaker_wav = np.array(speaker_samples)
                embedding = encoder.embed_utterance(speaker_wav)
                
                # Compare against stored profiles
                best_match = None
                best_similarity = 0.0
                
                for profile in stored_profiles:
                    stored_embedding = np.frombuffer(profile['embedding'], dtype=np.float32)
                    similarity = float(np.dot(embedding, stored_embedding))
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = profile
                
                if best_match and best_similarity >= match_threshold:
                    matched_name = best_match['speaker_name']
                    speaker_remap[speaker_name] = matched_name
                    matches.append({
                        "original_label": speaker_name,
                        "matched_to": matched_name,
                        "similarity": round(best_similarity, 3),
                    })
                    logger.info(f"Matched '{speaker_name}' to stored profile '{matched_name}' ({best_similarity:.0%})")
                    
            except Exception as e:
                logger.debug(f"Failed to match speaker '{speaker_name}': {e}")
        
        # Update segments with matched names
        updated_segments = []
        for seg in segments:
            new_seg = seg.copy()
            old_speaker = seg.get('speaker')
            if old_speaker in speaker_remap:
                new_seg['speaker'] = speaker_remap[old_speaker]
                new_seg['speaker_matched'] = True
            updated_segments.append(new_seg)
        
        return updated_segments, {
            "profiles_checked": len(stored_profiles),
            "matches": matches,
        }
        
    finally:
        if needs_cleanup and wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
