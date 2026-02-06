#!/usr/bin/env python3
"""
YouTube ML Pipeline for SpeechScore

Scrapes presentations from YouTube, extracts audio, runs through pipeline,
and trains a regression model to predict engagement from speech metrics.

Usage: python youtube_ml_pipeline.py [--scrape] [--download] [--analyze] [--train]
"""

import subprocess
import json
import os
import sys
import time
import gc
import argparse
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "youtube_data"
AUDIO_DIR = OUTPUT_DIR / "audio"
METADATA_FILE = OUTPUT_DIR / "metadata.json"
ANALYSIS_FILE = OUTPUT_DIR / "analyses.json"
MODEL_FILE = OUTPUT_DIR / "engagement_model.json"


@dataclass
class VideoData:
    video_id: str
    url: str
    title: str
    channel: str
    views: int
    likes: int
    duration: int
    like_ratio: float
    quality_score: float
    audio_path: Optional[str] = None
    analyzed: bool = False
    analysis: Optional[Dict] = None


def get_video_metadata(url: str) -> Optional[VideoData]:
    """Extract metadata from a YouTube video."""
    cmd = ["yt-dlp", "--dump-json", "--no-download", "--no-warnings", url]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None
        
        data = json.loads(result.stdout)
        views = data.get('view_count', 0) or 0
        likes = data.get('like_count', 0) or 0
        duration = data.get('duration', 0) or 0
        
        like_ratio = (likes / views * 100) if views > 0 else 0
        
        # Quality score: combines reach and approval
        view_score = math.log10(max(views, 1))
        ratio_score = min(like_ratio / 5.0, 1.0) * 10
        quality_score = (view_score * 0.4 + ratio_score * 0.6)
        
        return VideoData(
            video_id=data.get('id', ''),
            url=url,
            title=data.get('title', 'Unknown'),
            channel=data.get('channel', 'Unknown'),
            views=views,
            likes=likes,
            duration=duration,
            like_ratio=like_ratio,
            quality_score=quality_score,
        )
    except Exception as e:
        logger.error(f"Metadata error for {url}: {e}")
        return None


def search_youtube(query: str, max_results: int = 20) -> List[str]:
    """Search YouTube and return video URLs."""
    cmd = [
        "yt-dlp", "--flat-playlist", "--dump-json", "--no-warnings",
        f"ytsearch{max_results}:{query}",
    ]
    
    urls = []
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        for line in result.stdout.strip().split('\n'):
            if line:
                try:
                    data = json.loads(line)
                    if data.get('id'):
                        urls.append(f"https://www.youtube.com/watch?v={data['id']}")
                except:
                    pass
    except Exception as e:
        logger.error(f"Search error: {e}")
    
    return urls


def download_audio(video: VideoData, clip_duration: int = 180) -> bool:
    """Download audio clip from video."""
    safe_name = video.video_id
    output_path = AUDIO_DIR / f"{safe_name}.mp3"
    
    if output_path.exists():
        video.audio_path = str(output_path)
        return True
    
    cmd = [
        "yt-dlp",
        "-x", "--audio-format", "mp3",
        "--audio-quality", "5",
        "-o", str(output_path),
        "--no-warnings",
        "--postprocessor-args", f"ffmpeg:-t {clip_duration}",
        video.url
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if output_path.exists():
            video.audio_path = str(output_path)
            return True
        # Check for .mp3.mp3 (yt-dlp quirk)
        alt_path = str(output_path) + ".mp3"
        if os.path.exists(alt_path):
            os.rename(alt_path, str(output_path))
            video.audio_path = str(output_path)
            return True
    except Exception as e:
        logger.error(f"Download error: {e}")
    
    return False


def analyze_audio(video: VideoData) -> bool:
    """Run speech analysis pipeline on audio."""
    if not video.audio_path or not os.path.exists(video.audio_path):
        return False
    
    try:
        from api.pipeline_runner import run_pipeline, convert_to_wav
        
        wav_path = convert_to_wav(video.audio_path)
        
        result = run_pipeline(
            wav_path,
            modules=None,
            language="en",
            quality_gate="skip",  # Don't reject low quality
            preset="full",
            modules_requested=["all"],
        )
        
        video.analysis = result
        video.analyzed = True
        
        # Cleanup wav
        if wav_path != video.audio_path and os.path.exists(wav_path):
            os.unlink(wav_path)
        
        return True
    except Exception as e:
        logger.error(f"Analysis error for {video.video_id}: {e}")
        return False


def train_engagement_model(videos: List[VideoData]) -> Dict:
    """Train regression model to predict engagement from speech metrics."""
    import numpy as np
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    
    # Metrics to use as features
    FEATURE_METRICS = [
        'wpm', 'pitch_mean_hz', 'pitch_std_hz', 'pitch_range_hz',
        'jitter_pct', 'shimmer_pct', 'hnr_db', 'articulation_rate',
        'lexical_diversity', 'syntactic_complexity', 'fk_grade_level',
        'repetition_score', 'snr_db', 'vocal_fry_ratio', 'connectedness',
    ]
    
    # Build feature matrix
    X = []
    y = []
    valid_videos = []
    
    for v in videos:
        if not v.analyzed or not v.analysis:
            continue
        
        features = []
        valid = True
        for m in FEATURE_METRICS:
            val = v.analysis.get(m)
            if val is None:
                valid = False
                break
            features.append(float(val))
        
        if valid:
            X.append(features)
            y.append(v.quality_score)
            valid_videos.append(v)
    
    if len(X) < 10:
        logger.error(f"Not enough data: {len(X)} samples")
        return {"error": "Insufficient data"}
    
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Training on {len(X)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Try multiple models
    models = {
        'ridge': Ridge(alpha=1.0),
        'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'gradient_boost': GradientBoostingRegressor(n_estimators=50, max_depth=3),
        'random_forest': RandomForestRegressor(n_estimators=50, max_depth=4),
    }
    
    results = {}
    best_model = None
    best_score = -999
    
    for name, model in models.items():
        try:
            scores = cross_val_score(model, X_scaled, y, cv=min(5, len(X)//2), scoring='r2')
            mean_score = scores.mean()
            results[name] = {
                'r2_mean': float(mean_score),
                'r2_std': float(scores.std()),
            }
            logger.info(f"  {name}: R² = {mean_score:.3f} (+/- {scores.std():.3f})")
            
            if mean_score > best_score:
                best_score = mean_score
                best_model = (name, model)
        except Exception as e:
            logger.error(f"  {name} failed: {e}")
    
    # Train best model on full data
    if best_model:
        name, model = best_model
        model.fit(X_scaled, y)
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            importances = np.ones(len(FEATURE_METRICS)) / len(FEATURE_METRICS)
        
        # Normalize to weights
        weights = importances / importances.sum()
        
        # Create weight dict
        learned_weights = {
            FEATURE_METRICS[i]: float(weights[i])
            for i in range(len(FEATURE_METRICS))
        }
        
        # Sort by importance
        sorted_weights = dict(sorted(learned_weights.items(), key=lambda x: -x[1]))
        
        return {
            'best_model': name,
            'r2_score': float(best_score),
            'n_samples': len(X),
            'model_results': results,
            'learned_weights': sorted_weights,
            'feature_means': {FEATURE_METRICS[i]: float(scaler.mean_[i]) for i in range(len(FEATURE_METRICS))},
            'feature_stds': {FEATURE_METRICS[i]: float(scaler.scale_[i]) for i in range(len(FEATURE_METRICS))},
        }
    
    return {"error": "No model succeeded"}


def save_data(videos: List[VideoData]):
    """Save video data to JSON."""
    data = [asdict(v) for v in videos]
    with open(METADATA_FILE, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_data() -> List[VideoData]:
    """Load video data from JSON."""
    if not METADATA_FILE.exists():
        return []
    
    with open(METADATA_FILE, 'r') as f:
        data = json.load(f)
    
    return [VideoData(**d) for d in data]


# Search queries for diverse presentations
SEARCH_QUERIES = [
    # High engagement (likely good presentations)
    "TED talk most viewed",
    "best keynote speech ever",
    "viral motivational speech",
    "famous commencement address",
    "inspiring public speaking",
    
    # Medium engagement (average presentations)
    "business presentation example",
    "academic lecture recorded",
    "conference talk 2023",
    "workshop presentation",
    "seminar recording",
    
    # Diverse topics
    "science communication talk",
    "leadership speech",
    "startup pitch presentation",
    "educational lecture",
]


def main():
    parser = argparse.ArgumentParser(description='YouTube ML Pipeline')
    parser.add_argument('--scrape', action='store_true', help='Scrape video metadata')
    parser.add_argument('--download', action='store_true', help='Download audio clips')
    parser.add_argument('--analyze', action='store_true', help='Run speech analysis')
    parser.add_argument('--train', action='store_true', help='Train ML model')
    parser.add_argument('--all', action='store_true', help='Run full pipeline')
    parser.add_argument('--max-videos', type=int, default=100, help='Max videos to process')
    
    args = parser.parse_args()
    
    if args.all:
        args.scrape = args.download = args.analyze = args.train = True
    
    # Create directories
    OUTPUT_DIR.mkdir(exist_ok=True)
    AUDIO_DIR.mkdir(exist_ok=True)
    
    # Load existing data
    videos = load_data()
    existing_ids = {v.video_id for v in videos}
    
    # SCRAPE
    if args.scrape:
        logger.info("=" * 60)
        logger.info("PHASE 1: Scraping YouTube metadata")
        logger.info("=" * 60)
        
        all_urls = []
        for query in SEARCH_QUERIES:
            logger.info(f"Searching: {query}")
            urls = search_youtube(query, max_results=15)
            all_urls.extend(urls)
            time.sleep(1)
        
        # Dedupe
        all_urls = list(set(all_urls))
        logger.info(f"Found {len(all_urls)} unique URLs")
        
        # Get metadata
        new_count = 0
        for i, url in enumerate(all_urls):
            if new_count >= args.max_videos:
                break
            
            # Extract video ID
            vid = url.split('v=')[-1].split('&')[0]
            if vid in existing_ids:
                continue
            
            logger.info(f"[{i+1}/{len(all_urls)}] Getting metadata...")
            meta = get_video_metadata(url)
            
            if meta and meta.duration >= 120:  # At least 2 min
                videos.append(meta)
                existing_ids.add(meta.video_id)
                new_count += 1
                logger.info(f"  ✓ {meta.title[:40]} | {meta.views:,} views | Q={meta.quality_score:.2f}")
            
            time.sleep(0.5)
        
        save_data(videos)
        logger.info(f"Total videos: {len(videos)}")
    
    # DOWNLOAD
    if args.download:
        logger.info("=" * 60)
        logger.info("PHASE 2: Downloading audio clips")
        logger.info("=" * 60)
        
        to_download = [v for v in videos if not v.audio_path or not os.path.exists(v.audio_path or '')]
        logger.info(f"Need to download: {len(to_download)}")
        
        for i, video in enumerate(to_download[:args.max_videos]):
            logger.info(f"[{i+1}/{len(to_download)}] Downloading {video.title[:40]}...")
            
            if download_audio(video):
                logger.info(f"  ✓ Downloaded")
            else:
                logger.info(f"  ✗ Failed")
            
            time.sleep(1)
        
        save_data(videos)
    
    # ANALYZE
    if args.analyze:
        logger.info("=" * 60)
        logger.info("PHASE 3: Running speech analysis")
        logger.info("=" * 60)
        
        # Preload models
        logger.info("Loading pipeline models...")
        from api.pipeline_runner import preload_models
        preload_models()
        
        to_analyze = [v for v in videos if v.audio_path and not v.analyzed]
        logger.info(f"Need to analyze: {len(to_analyze)}")
        
        for i, video in enumerate(to_analyze[:args.max_videos]):
            logger.info(f"[{i+1}/{len(to_analyze)}] Analyzing {video.title[:40]}...")
            
            if analyze_audio(video):
                logger.info(f"  ✓ Analyzed")
            else:
                logger.info(f"  ✗ Failed")
            
            # Memory cleanup
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass
            
            # Save periodically
            if (i + 1) % 5 == 0:
                save_data(videos)
        
        save_data(videos)
    
    # TRAIN
    if args.train:
        logger.info("=" * 60)
        logger.info("PHASE 4: Training engagement model")
        logger.info("=" * 60)
        
        analyzed = [v for v in videos if v.analyzed]
        logger.info(f"Analyzed videos: {len(analyzed)}")
        
        if len(analyzed) >= 10:
            results = train_engagement_model(analyzed)
            
            with open(MODEL_FILE, 'w') as f:
                json.dump(results, f, indent=2)
            
            if 'learned_weights' in results:
                logger.info("\n" + "=" * 60)
                logger.info("LEARNED METRIC WEIGHTS (for presentation effectiveness)")
                logger.info("=" * 60)
                for metric, weight in results['learned_weights'].items():
                    bar = "█" * int(weight * 100)
                    logger.info(f"  {metric:25} {weight:.3f} {bar}")
                
                logger.info(f"\nModel R² score: {results['r2_score']:.3f}")
                logger.info(f"Saved to: {MODEL_FILE}")
        else:
            logger.error("Not enough analyzed videos for training")
    
    logger.info("\nDone!")


if __name__ == "__main__":
    main()
