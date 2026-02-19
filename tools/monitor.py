#!/usr/bin/env python3
"""
SpeakFit API Monitor

Checks API health and metrics, alerts via Discord webhook if issues detected.
Run via cron every 5 minutes:
  */5 * * * * /home/melchior/speech3/venv/bin/python /home/melchior/speech3/tools/monitor.py

Environment variables:
  SPEAKFIT_API_KEY      - Admin API key
  SPEAKFIT_API_URL      - API base URL (default: http://localhost:8000)
  DISCORD_WEBHOOK_URL   - Discord webhook for alerts
"""

import os
import sys
import json
import time
import urllib.request
import urllib.error
from datetime import datetime

# Config
API_URL = os.getenv("SPEAKFIT_API_URL", "http://localhost:8000")
API_KEY = os.getenv("SPEAKFIT_API_KEY", "sk-5b7fd66025ead6b8731ef73b2c970f26")
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL", "")

# Alert thresholds
THRESHOLDS = {
    "p95_latency_ms": 5000,       # Alert if p95 > 5s
    "error_rate_pct": 5.0,        # Alert if error rate > 5%
    "queue_depth": 15,            # Alert if queue > 15
    "pipeline_error_rate_pct": 10.0,  # Alert if pipeline errors > 10%
}

STATE_FILE = "/tmp/speakfit_monitor_state.json"


def load_state():
    """Load previous alert state."""
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except:
        return {"last_alert": 0, "alert_count": 0}


def save_state(state):
    """Save alert state."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def send_discord_alert(title: str, message: str, color: int = 0xFF0000):
    """Send alert to Discord webhook."""
    if not DISCORD_WEBHOOK:
        print(f"[ALERT] {title}: {message}")
        return
    
    payload = {
        "embeds": [{
            "title": f"🚨 {title}",
            "description": message,
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "SpeakFit Monitor"}
        }]
    }
    
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        DISCORD_WEBHOOK,
        data=data,
        headers={"Content-Type": "application/json"}
    )
    
    try:
        urllib.request.urlopen(req, timeout=10)
        print(f"[DISCORD] Alert sent: {title}")
    except Exception as e:
        print(f"[ERROR] Failed to send Discord alert: {e}")


def check_health():
    """Check basic API health."""
    url = f"{API_URL}/v1/health"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return {
                "healthy": data.get("status") == "ok",
                "gpu": data.get("gpu_available", False),
                "whisper": data.get("whisper_model_loaded", False),
                "uptime": data.get("uptime_sec", 0),
            }
    except Exception as e:
        return {"healthy": False, "error": str(e)}


def check_metrics():
    """Check detailed metrics."""
    url = f"{API_URL}/v1/metrics"
    try:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {API_KEY}"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}"}
    except Exception as e:
        return {"error": str(e)}


def analyze_metrics(metrics: dict) -> list:
    """Analyze metrics and return list of issues."""
    issues = []
    
    if "error" in metrics:
        issues.append(f"Cannot fetch metrics: {metrics['error']}")
        return issues
    
    requests = metrics.get("requests", {})
    pipeline = metrics.get("pipeline", {})
    queue = metrics.get("queue", {})
    
    # Check latency
    p95 = requests.get("p95_latency_ms", 0)
    if p95 > THRESHOLDS["p95_latency_ms"]:
        issues.append(f"High latency: p95={p95:.0f}ms (threshold: {THRESHOLDS['p95_latency_ms']}ms)")
    
    # Check error rate
    err_rate = requests.get("error_rate_pct", 0)
    if err_rate > THRESHOLDS["error_rate_pct"]:
        issues.append(f"High error rate: {err_rate:.1f}% (threshold: {THRESHOLDS['error_rate_pct']}%)")
    
    # Check queue depth
    q_depth = queue.get("current_depth", 0)
    if q_depth > THRESHOLDS["queue_depth"]:
        issues.append(f"Queue backing up: {q_depth} jobs (threshold: {THRESHOLDS['queue_depth']})")
    
    # Check pipeline errors
    pipe_err = pipeline.get("error_rate_pct", 0)
    if pipe_err > THRESHOLDS["pipeline_error_rate_pct"]:
        issues.append(f"Pipeline errors high: {pipe_err:.1f}% (threshold: {THRESHOLDS['pipeline_error_rate_pct']}%)")
    
    return issues


def main():
    state = load_state()
    now = time.time()
    
    # Rate limit alerts (max 1 per 15 minutes)
    alert_cooldown = 900
    can_alert = (now - state.get("last_alert", 0)) > alert_cooldown
    
    print(f"[{datetime.now().isoformat()}] SpeakFit Monitor")
    
    # Check health
    health = check_health()
    if not health.get("healthy"):
        error = health.get("error", "Unknown error")
        print(f"[CRITICAL] API unhealthy: {error}")
        if can_alert:
            send_discord_alert("API Down", f"Health check failed: {error}")
            state["last_alert"] = now
            state["alert_count"] = state.get("alert_count", 0) + 1
            save_state(state)
        return 1
    
    print(f"  Health: OK (GPU: {health['gpu']}, Whisper: {health['whisper']}, Uptime: {health['uptime']:.0f}s)")
    
    # Check GPU/Whisper
    if not health.get("gpu"):
        print("[WARN] GPU not available")
        if can_alert:
            send_discord_alert("GPU Unavailable", "NVIDIA GPU not detected. Analysis will fail.", 0xFFA500)
            state["last_alert"] = now
            save_state(state)
    
    if not health.get("whisper"):
        print("[WARN] Whisper model not loaded")
        if can_alert:
            send_discord_alert("Whisper Not Loaded", "Speech recognition model not loaded. Restart API.", 0xFFA500)
            state["last_alert"] = now
            save_state(state)
    
    # Check metrics
    metrics = check_metrics()
    if "error" in metrics:
        print(f"[WARN] Cannot fetch metrics: {metrics['error']}")
    else:
        req = metrics.get("requests", {})
        queue = metrics.get("queue", {})
        pipe = metrics.get("pipeline", {})
        
        print(f"  Requests: {req.get('total', 0)} ({req.get('per_minute', 0):.1f}/min)")
        print(f"  Latency: avg={req.get('avg_latency_ms', 0):.0f}ms, p95={req.get('p95_latency_ms', 0):.0f}ms")
        print(f"  Errors: {req.get('error_rate_pct', 0):.1f}%")
        print(f"  Queue: {queue.get('current_depth', 0)}/{queue.get('max_depth', 20)}")
        print(f"  Pipeline: {pipe.get('total_processed', 0)} processed, {pipe.get('error_rate_pct', 0):.1f}% errors")
        
        issues = analyze_metrics(metrics)
        if issues:
            print(f"[ISSUES] {len(issues)} problems detected:")
            for issue in issues:
                print(f"  - {issue}")
            
            if can_alert:
                send_discord_alert(
                    "Performance Issues",
                    "\n".join(f"• {i}" for i in issues),
                    0xFFA500
                )
                state["last_alert"] = now
                state["alert_count"] = state.get("alert_count", 0) + 1
                save_state(state)
    
    print("[OK] Monitor complete")
    save_state(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
