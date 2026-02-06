#!/usr/bin/env python3
"""
YouTube Presentation Scraper for SpeechScore ML Training

Scrapes popular presentations from YouTube, extracts:
- Audio (first 3 minutes)
- Engagement metrics (views, likes) as quality labels
- Metadata (title, channel, duration)

Uses yt-dlp for robust extraction.
"""

import subprocess
import json
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class VideoMetadata:
    video_id: str
    title: str
    channel: str
    views: int
    likes: int
    duration: int
    like_ratio: float
    upload_date: str
    
    @property
    def quality_score(self) -> float:
        """
        Compute a quality score based on engagement.
        Combines views (reach) and like_ratio (approval).
        """
        # Log scale for views (1M views = 6, 10M = 7, etc.)
        import math
        view_score = math.log10(max(self.views, 1))
        
        # Like ratio typically 1-5% for good content
        ratio_score = min(self.like_ratio / 5.0, 1.0) * 10
        
        # Weighted combination
        return (view_score * 0.4 + ratio_score * 0.6)


def get_video_metadata(url: str) -> Optional[VideoMetadata]:
    """Extract metadata from a YouTube video."""
    cmd = ["yt-dlp", "--dump-json", "--no-download", url]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None
        
        data = json.loads(result.stdout)
        
        views = data.get('view_count', 0) or 0
        likes = data.get('like_count', 0) or 0
        
        return VideoMetadata(
            video_id=data.get('id', ''),
            title=data.get('title', 'Unknown'),
            channel=data.get('channel', 'Unknown'),
            views=views,
            likes=likes,
            duration=data.get('duration', 0) or 0,
            like_ratio=(likes / views * 100) if views > 0 else 0,
            upload_date=data.get('upload_date', ''),
        )
    except Exception as e:
        print(f"Error getting metadata: {e}")
        return None


def download_audio_clip(url: str, output_path: str, start: int = 0, duration: int = 180) -> bool:
    """Download audio clip from YouTube video."""
    cmd = [
        "yt-dlp",
        "-x",  # Extract audio
        "--audio-format", "mp3",
        "--audio-quality", "5",  # Medium quality (smaller files)
        "-o", output_path,
        "--postprocessor-args", f"ffmpeg:-ss {start} -t {duration}",
        url
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return os.path.exists(output_path) or os.path.exists(output_path.replace('.mp3', '.mp3.mp3'))
    except Exception as e:
        print(f"Error downloading: {e}")
        return False


def search_presentations(query: str, max_results: int = 10) -> List[str]:
    """Search YouTube for presentations matching query."""
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--dump-json",
        f"ytsearch{max_results}:{query}",
    ]
    
    urls = []
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        for line in result.stdout.strip().split('\n'):
            if line:
                data = json.loads(line)
                if data.get('id'):
                    urls.append(f"https://www.youtube.com/watch?v={data['id']}")
    except Exception as e:
        print(f"Search error: {e}")
    
    return urls


# Curated list of high-quality presentation playlists/channels
SEED_VIDEOS = [
    # TED Talks (most viewed)
    "https://www.youtube.com/watch?v=8d9GqQHfRnc",  # Simon Sinek - Start with Why (65M+)
    "https://www.youtube.com/watch?v=UF8uR6Z6KLc",  # Steve Jobs Stanford (48M+)
    "https://www.youtube.com/watch?v=arj7oStGLkU",  # Tim Urban - Procrastination (70M+)
    "https://www.youtube.com/watch?v=eIho2S0ZahI",  # Amy Cuddy - Body Language (24M+)
    "https://www.youtube.com/watch?v=iCvmsMzlF7o",  # Brené Brown - Vulnerability (20M+)
    "https://www.youtube.com/watch?v=H14bBuluwB8",  # Chimamanda - Single Story (30M+)
    "https://www.youtube.com/watch?v=qp0HIF3SfI4",  # Ken Robinson - Schools Kill Creativity (75M+)
    
    # Tech presentations
    "https://www.youtube.com/watch?v=x7X9w_GIm1s",  # Guy Kawasaki - Art of Start
    "https://www.youtube.com/watch?v=Unzc731iCUY",  # How Google Works
    
    # Motivational
    "https://www.youtube.com/watch?v=IdTMDpizis8",  # Jim Rohn
]

SEARCH_QUERIES = [
    "TED talk most viewed",
    "best keynote speech",
    "motivational speech viral",
    "commencement speech famous",
    "public speaking masterclass",
]


if __name__ == "__main__":
    print("YouTube Presentation Scraper")
    print("=" * 50)
    
    output_dir = Path(__file__).parent / "youtube_speeches"
    output_dir.mkdir(exist_ok=True)
    
    results = []
    
    # Process seed videos
    print(f"\nProcessing {len(SEED_VIDEOS)} seed videos...")
    
    for i, url in enumerate(SEED_VIDEOS[:5]):  # Start with 5
        print(f"\n[{i+1}/{len(SEED_VIDEOS[:5])}] {url}")
        
        meta = get_video_metadata(url)
        if not meta:
            print("  ❌ Failed to get metadata")
            continue
        
        print(f"  📺 {meta.title[:50]}")
        print(f"  👁️ {meta.views:,} views | 👍 {meta.like_ratio:.2f}%")
        print(f"  📊 Quality score: {meta.quality_score:.2f}")
        
        results.append({
            "url": url,
            "metadata": meta.__dict__,
            "quality_score": meta.quality_score,
        })
        
        time.sleep(1)  # Rate limiting
    
    # Save results
    output_file = output_dir / "video_metadata.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Saved {len(results)} video metadata to {output_file}")
    print(f"\nNext: Run with --download flag to get audio clips")
