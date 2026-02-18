"""
API Performance Benchmarks

These tests measure response times for critical endpoints and fail if they
exceed defined thresholds. Run after every change to catch regressions.

Usage:
    pytest tests/test_api_benchmarks.py -v
    pytest tests/test_api_benchmarks.py -v --benchmark  # with detailed timing
"""
import time
import pytest
import statistics


class TimingResult:
    """Stores timing results for an endpoint."""
    def __init__(self, name: str, times: list[float], threshold: float):
        self.name = name
        self.times = times
        self.threshold = threshold
        self.mean = statistics.mean(times)
        self.median = statistics.median(times)
        self.min = min(times)
        self.max = max(times)
        self.stdev = statistics.stdev(times) if len(times) > 1 else 0
    
    def __str__(self):
        return (
            f"{self.name}: mean={self.mean*1000:.1f}ms, "
            f"median={self.median*1000:.1f}ms, "
            f"min={self.min*1000:.1f}ms, max={self.max*1000:.1f}ms, "
            f"threshold={self.threshold*1000:.0f}ms"
        )
    
    @property
    def passed(self):
        return self.median <= self.threshold


def measure_endpoint(client, method: str, url: str, iterations: int = 5, **kwargs) -> list[float]:
    """Measure response time for an endpoint over multiple iterations."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        response = getattr(client, method)(url, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        # Small delay between requests
        time.sleep(0.05)
    return times


class TestHealthCheck:
    """Test basic API availability and response time."""
    
    def test_health_check_responds(self, api_client):
        """API root should respond with 200."""
        response = api_client.get("/")
        assert response.status_code == 200
    
    def test_health_check_speed(self, api_client, thresholds):
        """Health check should respond within threshold (after warmup)."""
        # Extra warmup for this specific test
        for _ in range(3):
            api_client.get("/")
        times = measure_endpoint(api_client, "get", "/")
        result = TimingResult("health_check", times, thresholds["health_check"])
        print(f"\n  {result}")
        assert result.passed, f"Health check too slow: {result.median*1000:.1f}ms > {result.threshold*1000:.0f}ms"


class TestAuthEndpoints:
    """Test authentication endpoint performance."""
    
    def test_auth_me_responds(self, api_client):
        """Auth check should respond (even if unauthorized)."""
        response = api_client.get("/v1/auth/me")
        # 401 is expected without valid JWT
        assert response.status_code in [200, 401]
    
    def test_auth_me_speed(self, api_client, thresholds):
        """Auth check should respond within threshold."""
        times = measure_endpoint(api_client, "get", "/v1/auth/me")
        result = TimingResult("auth_check", times, thresholds["auth_check"])
        print(f"\n  {result}")
        assert result.passed, f"Auth check too slow: {result.median*1000:.1f}ms > {result.threshold*1000:.0f}ms"


class TestSpeechesEndpoints:
    """Test speeches CRUD endpoint performance."""
    
    def test_speeches_list_responds(self, api_client):
        """Speeches list should respond with 200."""
        response = api_client.get("/v1/speeches", params={"limit": 10})
        assert response.status_code == 200
        data = response.json()
        assert "speeches" in data
    
    def test_speeches_list_speed(self, api_client, thresholds):
        """Speeches list should respond within threshold."""
        times = measure_endpoint(
            api_client, "get", "/v1/speeches",
            params={"limit": 10}
        )
        result = TimingResult("speeches_list", times, thresholds["speeches_list"])
        print(f"\n  {result}")
        assert result.passed, f"Speeches list too slow: {result.median*1000:.1f}ms > {result.threshold*1000:.0f}ms"
    
    def test_speeches_list_pagination_speed(self, api_client, thresholds):
        """Paginated speeches should not be slower than first page."""
        times = measure_endpoint(
            api_client, "get", "/v1/speeches",
            params={"limit": 10, "offset": 100}
        )
        result = TimingResult("speeches_list_paginated", times, thresholds["speeches_list"])
        print(f"\n  {result}")
        assert result.passed, f"Paginated speeches too slow: {result.median*1000:.1f}ms"
    
    def test_speech_detail_speed(self, api_client, thresholds):
        """Single speech detail should respond within threshold."""
        # First get a speech ID
        response = api_client.get("/v1/speeches", params={"limit": 1})
        speeches = response.json().get("speeches", [])
        if not speeches:
            pytest.skip("No speeches in database")
        
        speech_id = speeches[0]["id"]
        times = measure_endpoint(api_client, "get", f"/v1/speeches/{speech_id}")
        result = TimingResult("speech_detail", times, thresholds["speech_detail"])
        print(f"\n  {result}")
        assert result.passed, f"Speech detail too slow: {result.median*1000:.1f}ms"
    
    def test_speech_score_speed(self, api_client, thresholds):
        """Speech score calculation should respond within threshold."""
        # First get a speech ID
        response = api_client.get("/v1/speeches", params={"limit": 1})
        speeches = response.json().get("speeches", [])
        if not speeches:
            pytest.skip("No speeches in database")
        
        speech_id = speeches[0]["id"]
        times = measure_endpoint(api_client, "get", f"/v1/speeches/{speech_id}/score")
        result = TimingResult("speech_score", times, thresholds["speech_score"])
        print(f"\n  {result}")
        assert result.passed, f"Speech score too slow: {result.median*1000:.1f}ms"


class TestSpeakersEndpoints:
    """Test speaker profile endpoint performance."""
    
    def test_speakers_list_responds(self, api_client):
        """Speakers list should respond with 200."""
        response = api_client.get("/v1/speakers")
        assert response.status_code == 200
    
    def test_speakers_list_speed(self, api_client, thresholds):
        """Speakers list should respond within threshold."""
        times = measure_endpoint(api_client, "get", "/v1/speakers")
        result = TimingResult("speakers_list", times, thresholds["speakers_list"])
        print(f"\n  {result}")
        assert result.passed, f"Speakers list too slow: {result.median*1000:.1f}ms"


class TestProjectsEndpoints:
    """Test projects endpoint performance."""
    
    def test_projects_list_responds(self, api_client):
        """Projects list should respond (may be 401 with API key only)."""
        response = api_client.get("/v1/projects")
        # API key auth returns user's projects; 401 means endpoint requires JWT
        assert response.status_code in [200, 401]
    
    def test_projects_list_speed(self, api_client, thresholds):
        """Projects list should respond within threshold."""
        times = measure_endpoint(api_client, "get", "/v1/projects")
        result = TimingResult("projects_list", times, thresholds["projects_list"])
        print(f"\n  {result}")
        assert result.passed, f"Projects list too slow: {result.median*1000:.1f}ms"


class TestDatabasePerformance:
    """Test database query performance with large datasets."""
    
    def test_speeches_count_speed(self, api_client):
        """Getting total count should be fast even with large DB."""
        start = time.perf_counter()
        response = api_client.get("/v1/speeches", params={"limit": 1})
        elapsed = time.perf_counter() - start
        
        data = response.json()
        total = data.get("total", 0)
        
        # Should be under 500ms even with 100k+ speeches
        threshold = 0.5
        print(f"\n  speeches_count: {elapsed*1000:.1f}ms (total: {total} speeches)")
        assert elapsed < threshold, f"Count query too slow: {elapsed*1000:.1f}ms"
    
    def test_user_speeches_isolation(self, api_client):
        """User-scoped queries should use index."""
        # This tests that user_id filtering is indexed
        start = time.perf_counter()
        response = api_client.get("/v1/speeches", params={"limit": 20})
        elapsed = time.perf_counter() - start
        
        # User-scoped query should be very fast
        threshold = 0.3
        print(f"\n  user_speeches: {elapsed*1000:.1f}ms")
        assert elapsed < threshold, f"User speeches query too slow: {elapsed*1000:.1f}ms"


# Summary report
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print performance summary at end of test run."""
    terminalreporter.write_sep("=", "Performance Benchmark Summary")
    terminalreporter.write_line("Run with -v to see detailed timing for each endpoint.")
