#!/usr/bin/env python3
"""
Adaptive pronunciation batch processor.
Checks API load and runs batches when server is underutilized.

Usage: python adaptive_pronunciation_batch.py [--check-only] [--batch-size N]
"""

import argparse
import json
import sqlite3
import sys
import time
import requests
from datetime import datetime
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
DB_PATH = SCRIPT_DIR / 'speechscore.db'
STATS_FILE = SCRIPT_DIR / 'pronunciation_batch_stats.json'
AUDIO_BASE = SCRIPT_DIR / 'audio'
API_URL = "http://localhost:8000"

# Thresholds
CPU_THRESHOLD = 70  # Don't run if CPU > 70%
MEMORY_THRESHOLD = 80  # Don't run if memory > 80%
GPU_MEMORY_THRESHOLD = 18000  # Don't run if GPU memory > 18GB used (of 24GB)


def load_stats():
    """Load batch statistics."""
    if STATS_FILE.exists():
        with open(STATS_FILE) as f:
            return json.load(f)
    return {
        "batches_completed": 0,
        "total_recordings_processed": 0,
        "batch_history": [],
        "average_batch_time_seconds": None,
        "last_check": None
    }


def save_stats(stats):
    """Save batch statistics."""
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)


API_KEY = "sk-5b7fd66025ead6b8731ef73b2c970f26"

def check_api_health():
    """Check if API is responsive."""
    try:
        # Use pronunciation accents endpoint as health check
        resp = requests.get(
            f"{API_URL}/v1/speeches/pronunciation/accents", 
            headers={"x-api-key": API_KEY},
            timeout=5
        )
        return resp.status_code == 200
    except:
        return False


def check_system_load():
    """Check CPU, memory, and GPU utilization."""
    import subprocess
    
    result = {
        "cpu_percent": 0,
        "memory_percent": 0,
        "gpu_memory_used_mb": 0,
        "is_underutilized": False
    }
    
    # CPU load (1-minute average)
    try:
        with open('/proc/loadavg') as f:
            load1 = float(f.read().split()[0])
            # Assume 8 cores, convert to percentage
            result["cpu_percent"] = min(100, load1 * 100 / 8)
    except:
        result["cpu_percent"] = 50  # Assume moderate if can't read
    
    # Memory
    try:
        with open('/proc/meminfo') as f:
            meminfo = dict(line.split(':') for line in f.read().strip().split('\n') if ':' in line)
            total = int(meminfo['MemTotal'].strip().split()[0])
            available = int(meminfo['MemAvailable'].strip().split()[0])
            result["memory_percent"] = 100 * (1 - available / total)
    except:
        result["memory_percent"] = 50
    
    # GPU memory
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            timeout=5
        ).decode().strip()
        result["gpu_memory_used_mb"] = int(output.split('\n')[0])
    except:
        result["gpu_memory_used_mb"] = 0
    
    # Determine if underutilized
    result["is_underutilized"] = (
        result["cpu_percent"] < CPU_THRESHOLD and
        result["memory_percent"] < MEMORY_THRESHOLD and
        result["gpu_memory_used_mb"] < GPU_MEMORY_THRESHOLD
    )
    
    return result


def get_pending_count():
    """Count recordings needing pronunciation analysis."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) FROM speeches 
        WHERE source_file IS NOT NULL 
          AND source_file != ''
          AND pronunciation_analyzed_at IS NULL
    """)
    count = cursor.fetchone()[0]
    conn.close()
    return count


def run_batch(batch_size: int = 100):
    """Run a batch of pronunciation analysis."""
    import subprocess
    
    start_time = time.time()
    
    # Run the batch script
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / 'batch_pronunciation_cron.py'),
            '--limit', str(batch_size),
            '--time-limit', '30'  # 30 minute max per batch
        ],
        capture_output=True,
        text=True,
        cwd=str(SCRIPT_DIR)
    )
    
    elapsed = time.time() - start_time
    success = result.returncode == 0
    
    return {
        "success": success,
        "elapsed_seconds": elapsed,
        "batch_size": batch_size,
        "output": result.stdout[-500:] if result.stdout else "",
        "error": result.stderr[-200:] if result.stderr else ""
    }


def main():
    parser = argparse.ArgumentParser(description='Adaptive pronunciation batch processor')
    parser.add_argument('--check-only', action='store_true', help='Only check status, don\'t run')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size (default: 100)')
    parser.add_argument('--force', action='store_true', help='Run even if server is busy')
    args = parser.parse_args()
    
    stats = load_stats()
    stats["last_check"] = datetime.utcnow().isoformat()
    
    # Check API health
    api_healthy = check_api_health()
    if not api_healthy:
        print("❌ API not responding - skipping batch")
        save_stats(stats)
        return 1
    
    # Check system load
    load = check_system_load()
    print(f"📊 System Load:")
    print(f"   CPU: {load['cpu_percent']:.1f}%")
    print(f"   Memory: {load['memory_percent']:.1f}%")
    print(f"   GPU Memory: {load['gpu_memory_used_mb']} MB")
    print(f"   Underutilized: {'✅ Yes' if load['is_underutilized'] else '❌ No'}")
    
    # Check pending work
    pending = get_pending_count()
    print(f"📋 Pending recordings: {pending:,}")
    
    # Stats summary
    print(f"\n📈 Batch Stats:")
    print(f"   Batches completed: {stats['batches_completed']}")
    print(f"   Total processed: {stats['total_recordings_processed']:,}")
    if stats['average_batch_time_seconds']:
        print(f"   Avg batch time: {stats['average_batch_time_seconds']:.1f}s")
    
    if args.check_only:
        save_stats(stats)
        return 0
    
    # Decide whether to run
    if not load['is_underutilized'] and not args.force:
        print("\n⏸️  Server is busy - skipping batch")
        save_stats(stats)
        return 0
    
    if pending == 0:
        print("\n✅ No recordings pending - nothing to do")
        save_stats(stats)
        return 0
    
    # Run batch
    print(f"\n🚀 Running batch of {args.batch_size} recordings...")
    batch_result = run_batch(args.batch_size)
    
    if batch_result['success']:
        print(f"✅ Batch completed in {batch_result['elapsed_seconds']:.1f}s")
        
        # Update stats
        stats['batches_completed'] += 1
        stats['total_recordings_processed'] += args.batch_size
        
        # Track batch history (keep last 20)
        stats['batch_history'].append({
            "timestamp": datetime.utcnow().isoformat(),
            "batch_size": args.batch_size,
            "elapsed_seconds": batch_result['elapsed_seconds'],
            "load_before": load
        })
        stats['batch_history'] = stats['batch_history'][-20:]
        
        # Calculate average
        times = [b['elapsed_seconds'] for b in stats['batch_history']]
        stats['average_batch_time_seconds'] = sum(times) / len(times)
        
        print(f"📊 New avg batch time: {stats['average_batch_time_seconds']:.1f}s")
    else:
        print(f"❌ Batch failed: {batch_result['error']}")
    
    save_stats(stats)
    return 0 if batch_result['success'] else 1


if __name__ == "__main__":
    sys.exit(main())
