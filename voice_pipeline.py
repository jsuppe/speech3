#!/usr/bin/env python3
"""
Melchior Voice Pipeline
Transcribes audio with faster-whisper, analyzes with speech3, generates spectrogram.
Usage: python voice_pipeline.py <audio_file> [--output-dir /tmp]
"""
import sys, os, json, time, subprocess, shutil, argparse

def generate_spectrogram(audio_path, output_dir="/tmp"):
    """Generate spectrogram + mel + loudness visualization using songsee."""
    basename = os.path.splitext(os.path.basename(audio_path))[0]
    wav_path = os.path.join(output_dir, f"{basename}.wav")
    spectrogram_path = os.path.join(output_dir, f"{basename}_spectrogram.png")

    # Convert to WAV if needed
    if not audio_path.endswith(".wav"):
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, wav_path],
            capture_output=True, timeout=30
        )
    else:
        wav_path = audio_path

    # Generate spectrogram with songsee
    songsee = shutil.which("songsee")
    if not songsee:
        return None

    result = subprocess.run(
        [songsee, wav_path, "--viz", "spectrogram,mel,loudness",
         "--format", "png", "-o", spectrogram_path,
         "--width", "1600", "--height", "900"],
        capture_output=True, text=True, timeout=30
    )

    if os.path.isfile(spectrogram_path):
        # Clean up temp wav
        if wav_path != audio_path and os.path.isfile(wav_path):
            os.remove(wav_path)
        return spectrogram_path
    return None

def main():
    parser = argparse.ArgumentParser(description="Melchior Voice Pipeline")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--output-dir", default="/tmp", help="Directory for output files")
    parser.add_argument("--no-spectrogram", action="store_true", help="Skip spectrogram generation")
    args = parser.parse_args()

    audio_path = args.audio_file
    if not os.path.isfile(audio_path):
        print(json.dumps({"error": f"File not found: {audio_path}"}))
        sys.exit(1)

    # Suppress noisy logs
    import logging
    logging.disable(logging.WARNING)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    sys.path.insert(0, os.path.dirname(__file__))

    # --- Audio Quality Assessment ---
    from audio_quality import load_audio_as_pcm, assess_quality

    t0 = time.time()
    audio_pcm, sr = load_audio_as_pcm(audio_path)
    quality = assess_quality(audio_pcm, sr)
    quality_time = time.time() - t0

    # --- STT ---
    from faster_whisper import WhisperModel

    t1 = time.time()
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    load_time = time.time() - t1

    t2 = time.time()
    segments, info = model.transcribe(audio_path, beam_size=5, language="en",
                                       vad_filter=True, vad_parameters={"min_silence_duration_ms": 300})
    
    segment_list = []
    full_text_parts = []
    for seg in segments:
        segment_list.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip()
        })
        full_text_parts.append(seg.text.strip())
    
    transcript = " ".join(full_text_parts)
    transcribe_time = time.time() - t2

    # --- speech3 text analysis (skip if audio quality is unusable) ---
    from analysis3_module import analyze_transcript

    t3 = time.time()
    if quality["confidence"] == "UNUSABLE":
        analysis = {"metrics": {}, "key_terms": [], "skipped": "Audio quality too low for reliable analysis"}
    else:
        analysis = analyze_transcript(transcript) if transcript else {"metrics": {}, "key_terms": []}
        if quality["confidence"] == "LOW":
            analysis["warning"] = "Low audio quality — results may be unreliable. Consider re-recording in a quieter environment."
    analysis_time = time.time() - t3

    # --- Advanced text analysis ---
    from advanced_text_analysis import analyze_advanced_text

    t3t = time.time()
    if quality["confidence"] == "UNUSABLE":
        advanced_text = {"skipped": "Audio quality too low for reliable analysis"}
    else:
        advanced_text = analyze_advanced_text(transcript) if transcript else {}
    advanced_text_time = time.time() - t3t

    # --- Rhetorical analysis ---
    from rhetorical_analysis import analyze_rhetoric

    t3r = time.time()
    if quality["confidence"] == "UNUSABLE":
        rhetoric = {"skipped": "Audio quality too low for reliable analysis"}
    else:
        rhetoric = analyze_rhetoric(transcript, segment_list) if transcript else {}
    rhetoric_time = time.time() - t3r

    # --- Sentiment analysis ---
    from sentiment_analysis import analyze_sentiment

    t3a = time.time()
    if quality["confidence"] == "UNUSABLE":
        sentiment = {"skipped": "Audio quality too low for reliable analysis"}
    else:
        sentiment = analyze_sentiment(transcript, segment_list) if transcript else {"overall": {"label": "NEUTRAL"}, "per_segment": [], "emotional_arc": "flat"}
        if quality["confidence"] == "LOW":
            sentiment["warning"] = "Low audio quality — transcript errors may affect sentiment accuracy."
    sentiment_time = time.time() - t3a

    # --- Audio-level speech analysis ---
    from audio_analysis import analyze_audio

    t3b = time.time()
    audio_duration = segment_list[-1]["end"] if segment_list else 0
    if quality["confidence"] == "UNUSABLE":
        audio_analysis = {"skipped": "Audio quality too low for reliable analysis"}
    else:
        audio_analysis = analyze_audio(audio_path, segment_list, audio_duration)
        if quality["confidence"] == "LOW":
            audio_analysis["warning"] = "Low audio quality — audio metrics may be affected by background noise."
    audio_analysis_time = time.time() - t3b

    # --- Advanced audio analysis ---
    from advanced_audio_analysis import analyze_advanced_audio

    t3c = time.time()
    if quality["confidence"] == "UNUSABLE":
        advanced_audio = {"skipped": "Audio quality too low for reliable analysis"}
    else:
        advanced_audio = analyze_advanced_audio(audio_path, segment_list)
    advanced_audio_time = time.time() - t3c

    # --- Spectrogram ---
    spectrogram_path = None
    spectrogram_time = 0
    if not args.no_spectrogram:
        t4 = time.time()
        spectrogram_path = generate_spectrogram(audio_path, args.output_dir)
        spectrogram_time = time.time() - t4

    # --- Output ---
    result = {
        "audio_file": audio_path,
        "audio_duration_sec": round(segments._audio_duration if hasattr(segments, '_audio_duration') else (segment_list[-1]["end"] if segment_list else 0), 2),
        "audio_quality": quality,
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "transcript": transcript,
        "segments": segment_list,
        "speech3_text_analysis": analysis,
        "advanced_text_analysis": advanced_text,
        "rhetorical_analysis": rhetoric,
        "sentiment": sentiment,
        "audio_analysis": audio_analysis,
        "advanced_audio_analysis": advanced_audio,
        "spectrogram": spectrogram_path,
        "timing": {
            "quality_check_sec": round(quality_time, 2),
            "model_load_sec": round(load_time, 2),
            "transcription_sec": round(transcribe_time, 2),
            "text_analysis_sec": round(analysis_time, 2),
            "advanced_text_sec": round(advanced_text_time, 2),
            "rhetoric_sec": round(rhetoric_time, 2),
            "sentiment_sec": round(sentiment_time, 2),
            "audio_analysis_sec": round(audio_analysis_time, 2),
            "advanced_audio_sec": round(advanced_audio_time, 2),
            "spectrogram_sec": round(spectrogram_time, 2),
            "total_sec": round(time.time() - t0, 2)
        }
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
