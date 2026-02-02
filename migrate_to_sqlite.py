#!/usr/bin/env python3
"""
Migrate benchmark_db.json + audio clips to SQLite database.

Reads existing JSON DB, finds matching audio clips, converts to Opus,
and stores everything in speechscore.db.
"""

import json
import os
import sys
import glob
from pathlib import Path

# Add speech3 to path
sys.path.insert(0, str(Path(__file__).parent))
from speech_db import SpeechDB


SPEECH3_DIR = Path(__file__).parent
BENCHMARK_DB = SPEECH3_DIR / "benchmark_db.json"
TEST_SPEECHES = SPEECH3_DIR / "test_speeches"


def find_clip_file(speech: dict) -> str | None:
    """Try to find the audio clip file for a benchmark entry."""
    # Try the 'file' field first
    source_file = speech.get("file", "")
    if source_file:
        # Try exact path
        if os.path.exists(source_file):
            return source_file
        # Try in test_speeches/
        candidate = TEST_SPEECHES / Path(source_file).name
        if candidate.exists():
            return str(candidate)

    # Try matching by title keywords
    title = speech.get("title", speech.get("name", "")).lower()
    clips = glob.glob(str(TEST_SPEECHES / "*clip*"))

    # Build keyword mapping for known entries
    keyword_map = {
        "i have a dream": "mlk_dream_climax",
        "mountaintop": "mlk_mountaintop",
        "1984 dnc": "cuomo_1984_opening",
        "casual conversational": None,  # Jon's voice notes - may not have clip
        "sky zone": None,
        "inaugural address (opening)": "jfk_inaugural_clip",
        "declaration of war": "fdr_war_declaration_clip",
        "first radio address": "churchill_pm_clip",
        "farewell to baseball": "gehrig_farewell_clip",
        "on black power": "carmichael",
        "checkers speech": "nixon_checkers",
        "inaugural address": "reagan_inaugural",  # Reagan
        "address on the berlin wall": "reagan_berlin",
        "stanford commencement (opening)": "jobs_stanford",
        "stanford commencement (connecting": "jobs_stanford",
        "primary reader": "children_reading_clip",
        "peter pan": "peter_pan",
        "commencement address (1990)": "carl_sagan_commencement_clip",
        "aesop": "aesop_children_fables_clip",
        "great debate": "bahnsen_debate_clip",
        "mild aphasia": "mild_aphasia_clip",
        "alice in wonderland": "alice_wonderland_clip",
        "kennedy-nixon": "kennedy_nixon_debate_clip",
        "chomsky": "chomsky_commencement_clip",
        "secret garden": "secret_garden_clip",
        "wizard of oz": "wizard_oz_clip",
        "psychology lecture": "psych_lecture_clip",
        "dysarthric speech - before": "dysarthric_before",
        "dysarthric speech - after": "dysarthric_after",
        "king george": "king_george_vi_clip",
        "gianna jessen": "gianna_jessen_clip",
        "pd avengers": "parkinsons_pd_avengers_clip",
        "hpr": "hpr_speech_impediments_clip",
        "spectrum": "autism_spectrum_clip",
        "pinocchio": "pinocchio_ch1_clip",
        "wind in the willows": "wind_willows_ch1_clip",
        "children's six minutes": "childrens_six_minutes_clip",
    }

    for keyword, clip_pattern in keyword_map.items():
        if keyword in title:
            if clip_pattern is None:
                return None
            # Search for matching clip
            for f in glob.glob(str(TEST_SPEECHES / "*")):
                if clip_pattern in f.lower() and f.endswith(".mp3"):
                    return f
            break

    # Fallback: search all clips for title words
    title_words = [w for w in title.split() if len(w) > 3]
    for clip in clips:
        clip_lower = clip.lower()
        if any(w in clip_lower for w in title_words):
            return clip

    return None


def build_api_result_from_legacy(speech: dict) -> dict:
    """
    Convert legacy benchmark_db metrics to API-like result format.
    This handles both old-style (flat metrics) and new-style (structured) entries.
    """
    # New-style entries already have structured data
    if "text_metrics" in speech:
        tm = speech.get("text_metrics", {})
        at = speech.get("advanced_text", {})
        am = speech.get("audio_metrics", {})
        adv_a = speech.get("advanced_audio", {})
        rhet = speech.get("rhetorical", {})
        sent = speech.get("sentiment", {})
        aq = speech.get("audio_quality", {})

        return {
            "transcription": {
                "text": speech.get("transcript_excerpt", ""),
                "segments": [],
            },
            "audio_quality": {
                "quality_level": speech.get("quality_tier", "MEDIUM"),
                "snr_db": aq.get("snr_db"),
                "confidence": aq.get("confidence"),
            },
            "audio_analysis": {
                "speaking_rate": {
                    "words_per_minute": am.get("speaking_rate_wpm"),
                },
                "voice_quality": {
                    "jitter_percent": am.get("jitter_percent"),
                    "shimmer_percent": am.get("shimmer_percent"),
                    "hnr_db": am.get("hnr_db"),
                },
                "pitch": {
                    "mean_f0": am.get("mean_pitch_hz"),
                    "std_f0": am.get("pitch_std_hz"),
                    "range_f0": am.get("pitch_range_hz"),
                },
            },
            "advanced_audio_analysis": {
                "vocal_fry": {
                    "vocal_fry_ratio": adv_a.get("vocal_fry_ratio"),
                    "detected": adv_a.get("vocal_fry_detected"),
                },
                "uptalk": {
                    "uptalk_ratio": adv_a.get("uptalk_ratio"),
                },
                "rate_variability": {
                    "cv": adv_a.get("rate_variability_cv"),
                },
                "formants": {
                    "f1_mean": adv_a.get("f1_mean_hz"),
                    "f2_mean": adv_a.get("f2_mean_hz"),
                    "vowel_space_index": adv_a.get("vowel_space_index"),
                    "articulation_clarity": adv_a.get("articulation_clarity"),
                },
            },
            "speech3_text_analysis": {
                "metrics": {
                    "lexical_diversity": tm.get("lexical_diversity"),
                    "syntactic_complexity": tm.get("syntactic_complexity"),
                    "fluency_score": tm.get("fluency_score"),
                    "connectedness": tm.get("connectedness"),
                    "mlu": tm.get("mlu"),
                },
            },
            "advanced_text_analysis": {
                "readability": {
                    "flesch_reading_ease": at.get("flesch_reading_ease"),
                    "flesch_kincaid_grade": at.get("flesch_kincaid_grade"),
                },
            },
            "rhetorical_analysis": {
                "repetition": {
                    "repetition_score": rhet.get("repetition_score"),
                },
                "anaphora": {
                    "count": rhet.get("anaphora_count"),
                },
                "rhythm": {
                    "regularity": rhet.get("rhythm_regularity"),
                },
            },
            "sentiment": {
                "overall": sent.get("overall", {}),
            },
            "timing": speech.get("timing", {}),
        }

    # Old-style entries with flat metrics
    m = speech.get("metrics", {})
    return {
        "transcription": {
            "text": speech.get("transcript_excerpt", ""),
            "segments": [],
        },
        "audio_quality": {
            "quality_level": speech.get("quality_tier", "MEDIUM"),
        },
        "audio_analysis": {
            "speaking_rate": {
                "words_per_minute": m.get("speaking_rate_wpm", m.get("wpm")),
            },
            "voice_quality": {
                "jitter_percent": m.get("jitter_percent"),
                "shimmer_percent": m.get("shimmer_percent"),
                "hnr_db": m.get("harmonics_to_noise_db", m.get("hnr_db")),
            },
            "pitch": {
                "mean_f0": m.get("pitch_mean_hz", m.get("mean_pitch_hz")),
                "range_f0": m.get("pitch_range_hz"),
                "std_f0": m.get("pitch_std_hz"),
            },
        },
        "advanced_audio_analysis": {
            "vocal_fry": {
                "vocal_fry_ratio": m.get("vocal_fry_percent", m.get("vocal_fry_ratio")),
            },
        },
        "speech3_text_analysis": {
            "metrics": {
                "lexical_diversity": m.get("lexical_diversity"),
                "syntactic_complexity": m.get("syntactic_complexity"),
                "connectedness": m.get("connectedness"),
            },
        },
        "rhetorical_analysis": {
            "repetition": {
                "repetition_score": m.get("repetition_score"),
            },
        },
        "sentiment": {},
        "timing": {},
    }


def migrate():
    """Run the migration."""
    print(f"Loading benchmark DB from {BENCHMARK_DB}")
    with open(BENCHMARK_DB) as f:
        data = json.load(f)

    speeches = data.get("speeches", [])
    print(f"Found {len(speeches)} speeches to migrate")

    db = SpeechDB()
    migrated = 0
    audio_stored = 0
    skipped = 0

    for i, speech in enumerate(speeches):
        title = speech.get("title", speech.get("name", f"Unknown #{i}"))
        speaker = speech.get("speaker", "")
        year = speech.get("year")
        category = speech.get("category", "")
        products = speech.get("products", [])
        source = speech.get("source", "")
        description = speech.get("description", "")

        print(f"\n[{i+1}/{len(speeches)}] {title}")

        # Find audio clip
        clip_path = find_clip_file(speech)

        # Build API-style result from legacy data
        api_result = build_api_result_from_legacy(speech)

        # Get audio hash for dedup
        audio_hash = None
        duration = None
        if clip_path:
            audio_hash = SpeechDB.get_audio_hash(clip_path)
            duration = SpeechDB.get_audio_duration(clip_path)

            # Check for duplicate
            existing = db.find_speech_by_hash(audio_hash)
            if existing:
                print(f"  ⚠ Duplicate (matches speech_id={existing['id']}), skipping")
                skipped += 1
                continue

        # Create speech record
        speech_id = db.add_speech(
            title=title,
            speaker=speaker,
            year=year,
            source_url=source if source.startswith("http") else None,
            source_file=speech.get("file", clip_path),
            duration_sec=duration,
            category=category,
            products=products if products else ["SpeechScore"],
            description=description,
            audio_hash=audio_hash,
        )

        # Store audio as Opus
        if clip_path and os.path.exists(clip_path):
            try:
                db.store_audio(speech_id, clip_path)
                audio_stored += 1
                size_kb = os.path.getsize(clip_path) / 1024
                print(f"  ✓ Audio: {Path(clip_path).name} ({size_kb:.0f} KB → Opus)")
            except Exception as e:
                print(f"  ✗ Audio encoding failed: {e}")
        else:
            print(f"  - No audio clip found")

        # Store transcription
        tx = api_result.get("transcription", {})
        if tx.get("text"):
            db.store_transcription(speech_id, tx["text"], tx.get("segments"))

        # Store analysis
        db.store_analysis(speech_id, api_result, preset="migrated")

        # Check for spectrogram
        # Look for spectrogram files matching the speech
        for spec_dir in [SPEECH3_DIR / "spectrograms", TEST_SPEECHES]:
            for spec_ext in ["*.png", "*.jpg"]:
                for spec_file in glob.glob(str(spec_dir / spec_ext)):
                    spec_name = Path(spec_file).stem.lower()
                    title_lower = title.lower()
                    if any(w in spec_name for w in title_lower.split() if len(w) > 3):
                        try:
                            db.store_spectrogram_from_file(speech_id, spec_file)
                            print(f"  ✓ Spectrogram: {Path(spec_file).name}")
                        except Exception as e:
                            print(f"  ✗ Spectrogram failed: {e}")
                        break

        migrated += 1

    print(f"\n{'='*60}")
    print(f"Migration complete!")
    print(f"  Migrated: {migrated}")
    print(f"  Audio stored: {audio_stored}")
    print(f"  Skipped (duplicates): {skipped}")
    print()

    # Print stats
    stats = db.stats()
    print(f"Database stats:")
    print(f"  Speeches:     {stats['speeches']}")
    print(f"  Analyses:     {stats['analyses']}")
    print(f"  Audio files:  {stats['audio_files']} ({stats['audio_size_mb']} MB)")
    print(f"  Spectrograms: {stats['spectrograms']}")
    print(f"  DB size:      {stats['db_size_mb']} MB")
    print(f"  Products:     {stats['products']}")

    db.close()


if __name__ == "__main__":
    migrate()
