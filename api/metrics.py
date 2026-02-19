"""
Metrics collection for SpeechScore API monitoring.

Tracks response times, queue depth, error rates, GPU usage.
"""

import time
import threading
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger("speechscore.metrics")


@dataclass
class RequestMetric:
    timestamp: float
    endpoint: str
    method: str
    status_code: int
    duration_ms: float
    user_tier: str = "unknown"


class MetricsCollector:
    """Collects and aggregates API metrics."""
    
    def __init__(self, window_minutes: int = 15):
        self._lock = threading.Lock()
        self._window_sec = window_minutes * 60
        
        # Recent requests (sliding window)
        self._requests: deque = deque(maxlen=10000)
        
        # Pipeline metrics
        self._pipeline_durations: deque = deque(maxlen=100)
        self._pipeline_errors: int = 0
        self._pipeline_success: int = 0
        
        # Track slow requests (> 5s)
        self._slow_requests: deque = deque(maxlen=100)
        
        # Peak queue depth seen
        self._peak_queue_depth: int = 0
        self._peak_queue_time: Optional[str] = None
    
    def record_request(self, endpoint: str, method: str, status_code: int,
                       duration_ms: float, user_tier: str = "unknown"):
        """Record an API request."""
        now = time.time()
        metric = RequestMetric(
            timestamp=now,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            duration_ms=duration_ms,
            user_tier=user_tier,
        )
        
        with self._lock:
            self._requests.append(metric)
            
            # Track slow requests
            if duration_ms > 5000:
                self._slow_requests.append(metric)
    
    def record_pipeline(self, duration_sec: float, success: bool):
        """Record a pipeline execution."""
        with self._lock:
            self._pipeline_durations.append(duration_sec)
            if success:
                self._pipeline_success += 1
            else:
                self._pipeline_errors += 1
    
    def record_queue_depth(self, depth: int):
        """Track peak queue depth."""
        with self._lock:
            if depth > self._peak_queue_depth:
                self._peak_queue_depth = depth
                self._peak_queue_time = datetime.now(timezone.utc).isoformat()
    
    def _get_recent_requests(self) -> List[RequestMetric]:
        """Get requests within the sliding window."""
        cutoff = time.time() - self._window_sec
        with self._lock:
            return [r for r in self._requests if r.timestamp > cutoff]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics snapshot."""
        recent = self._get_recent_requests()
        
        with self._lock:
            pipeline_durations = list(self._pipeline_durations)
            pipeline_success = self._pipeline_success
            pipeline_errors = self._pipeline_errors
            slow_count = len([r for r in self._slow_requests 
                            if r.timestamp > time.time() - self._window_sec])
            peak_queue = self._peak_queue_depth
            peak_queue_time = self._peak_queue_time
        
        # Request stats
        total_requests = len(recent)
        if recent:
            durations = [r.duration_ms for r in recent]
            avg_latency = sum(durations) / len(durations)
            p50_latency = sorted(durations)[len(durations) // 2]
            p95_latency = sorted(durations)[int(len(durations) * 0.95)] if len(durations) >= 20 else max(durations)
            p99_latency = sorted(durations)[int(len(durations) * 0.99)] if len(durations) >= 100 else max(durations)
            
            error_count = sum(1 for r in recent if r.status_code >= 500)
            error_rate = error_count / total_requests * 100
            
            # Requests per minute
            window_min = self._window_sec / 60
            rpm = total_requests / window_min
        else:
            avg_latency = p50_latency = p95_latency = p99_latency = 0
            error_rate = 0
            rpm = 0
        
        # Pipeline stats
        if pipeline_durations:
            avg_pipeline = sum(pipeline_durations) / len(pipeline_durations)
            p95_pipeline = sorted(pipeline_durations)[int(len(pipeline_durations) * 0.95)]
        else:
            avg_pipeline = p95_pipeline = 0
        
        pipeline_total = pipeline_success + pipeline_errors
        pipeline_error_rate = (pipeline_errors / pipeline_total * 100) if pipeline_total > 0 else 0
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "window_minutes": self._window_sec // 60,
            
            # Request metrics
            "requests": {
                "total": total_requests,
                "per_minute": round(rpm, 2),
                "avg_latency_ms": round(avg_latency, 1),
                "p50_latency_ms": round(p50_latency, 1),
                "p95_latency_ms": round(p95_latency, 1),
                "p99_latency_ms": round(p99_latency, 1),
                "error_rate_pct": round(error_rate, 2),
                "slow_requests": slow_count,
            },
            
            # Pipeline metrics
            "pipeline": {
                "total_processed": pipeline_total,
                "success": pipeline_success,
                "errors": pipeline_errors,
                "error_rate_pct": round(pipeline_error_rate, 2),
                "avg_duration_sec": round(avg_pipeline, 2),
                "p95_duration_sec": round(p95_pipeline, 2),
            },
            
            # Queue metrics
            "queue": {
                "peak_depth": peak_queue,
                "peak_time": peak_queue_time,
            },
        }
    
    def reset_peaks(self):
        """Reset peak counters (call daily or as needed)."""
        with self._lock:
            self._peak_queue_depth = 0
            self._peak_queue_time = None


# Global singleton
metrics = MetricsCollector()
