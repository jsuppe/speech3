"""
Speaker diarization module using resemblyzer for speaker embeddings
and agglomerative clustering to identify distinct speakers.
"""

import os
import logging
import tempfile
import subprocess
import numpy as np
from typing import List, Dict, Optional

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
) -> List[Dict]:
    """
    Run speaker diarization on audio and assign speaker labels to segments.
    
    Args:
        audio_path: Path to audio file
        segments: List of transcript segments with 'start', 'end', 'text'
        num_speakers: Fixed number of speakers (None = auto-detect)
        min_segment_duration: Minimum segment duration to process (seconds)
    
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
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.7,
                linkage='average'
            )
        
        labels = clustering.fit_predict(embeddings_array)
        n_speakers = len(set(labels))
        
        logger.info(f"Diarization: detected {n_speakers} speakers in {len(segments)} segments")
        
        # Assign speaker labels to segments
        speaker_labels = {}
        for idx, label in zip(valid_indices, labels):
            speaker_labels[idx] = f"SPEAKER_{label}"
        
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
