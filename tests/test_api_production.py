"""
Production API Benchmarks

Tests against the production API (api.speakfit.app) to measure real-world
latency including SSL and Cloudflare tunnel overhead.

Usage:
    API_BASE_URL=https://api.speakfit.app pytest tests/test_api_production.py -v
"""
import os
import time
import pytest
import httpx

# Only run these tests when explicitly targeting production
pytestmark = pytest.mark.skipif(
    "speakfit.app" not in os.getenv("API_BASE_URL", ""),
    reason="Production tests require API_BASE_URL=https://api.speakfit.app"
)

API_KEY = os.getenv("API_KEY", "sk-5b7fd66025ead6b8731ef73b2c970f26")

# Production thresholds are higher due to network latency
PROD_THRESHOLDS = {
    "cold_start": 3.0,         # First request (SSL handshake, cold start)
    "warm_request": 0.5,       # Subsequent requests
    "speeches_list": 1.0,
    "speech_detail": 0.8,
}


@pytest.fixture(scope="module")
def prod_client():
    """HTTP client for production API."""
    return httpx.Client(
        base_url=os.getenv("API_BASE_URL"),
        timeout=30.0,
        headers={"X-API-Key": API_KEY}
    )


class TestProductionLatency:
    """Test real-world production latency."""
    
    def test_cold_start_latency(self, prod_client):
        """First request includes SSL handshake and potential cold start."""
        start = time.perf_counter()
        response = prod_client.get("/")
        cold_time = time.perf_counter() - start
        
        assert response.status_code == 200
        print(f"\n  Cold start: {cold_time*1000:.0f}ms")
        assert cold_time < PROD_THRESHOLDS["cold_start"], \
            f"Cold start too slow: {cold_time*1000:.0f}ms"
    
    def test_warm_request_latency(self, prod_client):
        """Subsequent requests should reuse connection."""
        # Warm up
        prod_client.get("/")
        
        times = []
        for _ in range(5):
            start = time.perf_counter()
            response = prod_client.get("/")
            times.append(time.perf_counter() - start)
            assert response.status_code == 200
        
        avg_time = sum(times) / len(times)
        print(f"\n  Warm requests: avg={avg_time*1000:.0f}ms, "
              f"min={min(times)*1000:.0f}ms, max={max(times)*1000:.0f}ms")
        assert avg_time < PROD_THRESHOLDS["warm_request"], \
            f"Warm requests too slow: {avg_time*1000:.0f}ms"
    
    def test_speeches_list_production(self, prod_client):
        """Speeches list in production."""
        # Warm up connection
        prod_client.get("/")
        
        start = time.perf_counter()
        response = prod_client.get("/v1/speeches", params={"limit": 10})
        elapsed = time.perf_counter() - start
        
        assert response.status_code == 200
        data = response.json()
        print(f"\n  Speeches list: {elapsed*1000:.0f}ms ({data.get('total', 0)} total)")
        assert elapsed < PROD_THRESHOLDS["speeches_list"], \
            f"Speeches list too slow: {elapsed*1000:.0f}ms"
    
    def test_full_app_flow_simulation(self, prod_client):
        """Simulate app startup flow: auth check -> speeches list."""
        times = {}
        
        # 1. Auth check (simulated - we use API key)
        start = time.perf_counter()
        prod_client.get("/v1/auth/me")
        times["auth"] = time.perf_counter() - start
        
        # 2. Load speeches
        start = time.perf_counter()
        response = prod_client.get("/v1/speeches", params={"limit": 20})
        times["speeches"] = time.perf_counter() - start
        
        # 3. Load first speech detail (if any)
        speeches = response.json().get("speeches", [])
        if speeches:
            start = time.perf_counter()
            prod_client.get(f"/v1/speeches/{speeches[0]['id']}")
            times["detail"] = time.perf_counter() - start
        
        total = sum(times.values())
        print(f"\n  App flow simulation:")
        for name, t in times.items():
            print(f"    {name}: {t*1000:.0f}ms")
        print(f"    TOTAL: {total*1000:.0f}ms")
        
        # Total app startup should be under 3 seconds
        assert total < 3.0, f"App flow too slow: {total*1000:.0f}ms"
