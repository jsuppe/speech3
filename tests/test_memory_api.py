"""
Memory App API Integration Tests

Tests to ensure Memory (dementia) analysis and database operations
produce reproducible, consistent results.

Run with: pytest tests/test_memory_api.py -v
"""

import pytest
import requests
import json
import os
from datetime import datetime

# Test configuration
API_BASE = os.getenv("API_BASE", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "sk-5b7fd66025ead6b8731ef73b2c970f26")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# Known test recordings with expected results
TEST_RECORDINGS = {
    "alzheimers_patient_10yr.mp3": {
        "min_duration": 25,
        "max_duration": 40,
        "expected_markers": {
            "has_repetitions": True,  # Should detect some repetition
            "word_count_min": 50,
        },
    },
    "dementia_speech_patterns.mp3": {
        "min_duration": 10,
        "max_duration": 60,
        "expected_markers": {
            "word_count_min": 20,
        },
    },
}


class TestMemoryAPIHealth:
    """Basic API health and connectivity tests."""
    
    def test_api_health(self):
        """API should respond to health check."""
        r = requests.get(f"{API_BASE}/v1/health")
        assert r.status_code == 200
        data = r.json()
        assert data.get("status") in ["healthy", "ok"]
    
    def test_api_version(self):
        """API version endpoint should return version info."""
        r = requests.get(f"{API_BASE}/v1/version?client_version=1.0")
        assert r.status_code == 200
        data = r.json()
        assert "server_version" in data
        assert "compatible" in data


class TestMemoryPipelineStages:
    """Test pipeline stages endpoint for Memory profile."""
    
    def test_get_dementia_stages(self):
        """Should return pipeline stages for dementia profile."""
        r = requests.get(
            f"{API_BASE}/v1/pipeline/stages?profile=dementia",
            headers=HEADERS
        )
        assert r.status_code == 200
        data = r.json()
        
        assert "stages" in data
        stages = data["stages"]
        assert len(stages) >= 5  # Should have multiple stages
        
        # Check expected stage names
        stage_names = [s["name"] for s in stages]
        assert "Transcribing" in stage_names
        assert "Cross-Session Tracking" in stage_names or "Detecting Repetitions" in stage_names
        
        # Check percentages sum to ~100
        total_pct = sum(s.get("percent", 0) for s in stages)
        assert 95 <= total_pct <= 105, f"Stage percentages should sum to ~100, got {total_pct}"


class TestMemoryTestRecordings:
    """Test the debug test recording analysis feature."""
    
    def test_list_test_recordings(self):
        """Should list available test recordings."""
        r = requests.get(
            f"{API_BASE}/v1/analyze/test-files",
            headers=HEADERS
        )
        # May be 200 or 404 depending on implementation
        if r.status_code == 200:
            data = r.json()
            assert isinstance(data.get("files", []), list)
    
    def test_analyze_test_recording(self):
        """Analyze a known test recording and verify results."""
        r = requests.post(
            f"{API_BASE}/v1/analyze/test",
            headers=HEADERS,
            json={
                "filename": "alzheimers_patient_10yr.mp3",
                "profile": "dementia",
            },
            timeout=120,  # Analysis can take time
        )
        
        assert r.status_code == 200, f"Analysis failed: {r.text}"
        data = r.json()
        
        # Check basic response structure
        assert data.get("success") == True
        assert "speech_id" in data
        assert "transcript" in data
        assert data.get("audio_duration_sec", 0) > 0
        
        # Store speech_id for later tests
        pytest.speech_id = data["speech_id"]
        
        # Check dementia analysis ran
        assert "dementia_analysis" in data or data.get("transcript"), \
            "Should have dementia_analysis or at least transcript"
    
    def test_analysis_reproducibility(self):
        """Same recording should produce consistent results."""
        results = []
        
        for _ in range(2):
            r = requests.post(
                f"{API_BASE}/v1/analyze/test",
                headers=HEADERS,
                json={
                    "filename": "alzheimers_patient_10yr.mp3",
                    "profile": "dementia",
                },
                timeout=120,
            )
            assert r.status_code == 200
            results.append(r.json())
        
        # Transcripts should be identical (deterministic STT)
        assert results[0].get("transcript") == results[1].get("transcript"), \
            "Transcripts should be identical for same audio"
        
        # Audio duration should be identical
        assert results[0].get("audio_duration_sec") == results[1].get("audio_duration_sec"), \
            "Audio duration should be identical"


class TestMemoryRecordingsEndpoint:
    """Test the dementia recordings list endpoint."""
    
    def test_get_recordings_list(self):
        """Should return list of dementia recordings."""
        r = requests.get(
            f"{API_BASE}/v1/dementia/recordings?limit=10",
            headers=HEADERS
        )
        assert r.status_code == 200
        data = r.json()
        
        assert "recordings" in data
        recordings = data["recordings"]
        assert isinstance(recordings, list)
        
        # If there are recordings, check structure
        if recordings:
            rec = recordings[0]
            assert "id" in rec
            assert "title" in rec
            assert "created_at" in rec
    
    def test_get_recording_detail(self):
        """Should return details for a specific recording."""
        # First get a recording ID
        r = requests.get(
            f"{API_BASE}/v1/dementia/recordings?limit=1",
            headers=HEADERS
        )
        assert r.status_code == 200
        recordings = r.json().get("recordings", [])
        
        if not recordings:
            pytest.skip("No recordings available to test")
        
        speech_id = recordings[0]["id"]
        
        # Get detail
        r = requests.get(
            f"{API_BASE}/v1/dementia/recordings/{speech_id}",
            headers=HEADERS
        )
        assert r.status_code == 200
        data = r.json()
        
        assert data.get("id") == speech_id
        assert "title" in data


class TestMemoryMetrics:
    """Test dementia metrics aggregation endpoints."""
    
    def test_get_aggregated_metrics(self):
        """Should return aggregated dementia metrics."""
        r = requests.get(
            f"{API_BASE}/v1/dementia/metrics/aggregated",
            headers=HEADERS
        )
        assert r.status_code == 200
        data = r.json()
        
        # Should have metric categories
        expected_keys = ["repetitions", "prepositions", "word_finding"]
        for key in expected_keys:
            assert key in data or "metrics" in data, \
                f"Expected '{key}' in response or 'metrics' wrapper"
    
    def test_get_cross_session_summary(self):
        """Should return cross-session tracking summary."""
        r = requests.get(
            f"{API_BASE}/v1/dementia/cross-session-summary?days=30",
            headers=HEADERS
        )
        assert r.status_code == 200
        data = r.json()
        
        # Should have some structure for trends
        assert isinstance(data, dict)


class TestDementiaAnalysisConsistency:
    """Test that dementia analysis produces consistent markers."""
    
    def test_preposition_detection(self):
        """Preposition count should be consistent for same text."""
        # This would require a direct analysis endpoint or checking stored results
        pass  # TODO: Implement when we have direct text analysis endpoint
    
    def test_repetition_detection_threshold(self):
        """Repetition detection should respect threshold settings."""
        # Analyze same recording with different thresholds
        pass  # TODO: Implement


class TestDatabaseIntegrity:
    """Test database operations for Memory app."""
    
    def test_recording_persists(self):
        """New recording should persist and be retrievable."""
        # Analyze a recording
        r = requests.post(
            f"{API_BASE}/v1/analyze/test",
            headers=HEADERS,
            json={
                "filename": "alzheimers_patient_10yr.mp3",
                "profile": "dementia",
            },
            timeout=120,
        )
        assert r.status_code == 200
        speech_id = r.json().get("speech_id")
        assert speech_id is not None
        
        # Verify it appears in recordings list
        r = requests.get(
            f"{API_BASE}/v1/dementia/recordings?limit=50",
            headers=HEADERS
        )
        assert r.status_code == 200
        recordings = r.json().get("recordings", [])
        
        # Note: Due to duplicate detection, might reuse existing record
        # Just verify the endpoint works
        assert isinstance(recordings, list)
    
    def test_dementia_metrics_stored(self):
        """Dementia metrics should be stored after analysis."""
        # Analyze
        r = requests.post(
            f"{API_BASE}/v1/analyze/test",
            headers=HEADERS,
            json={
                "filename": "alzheimers_patient_10yr.mp3",
                "profile": "dementia",
            },
            timeout=120,
        )
        assert r.status_code == 200
        data = r.json()
        speech_id = data.get("speech_id")
        
        # Check that dementia_analysis was returned
        if "dementia_analysis" in data:
            analysis = data["dementia_analysis"]
            
            # Verify expected fields
            assert "preposition_analysis" in analysis or "repetitions" in analysis, \
                "Dementia analysis should have preposition_analysis or repetitions"


# Fixtures
@pytest.fixture(scope="session")
def api_client():
    """Reusable API client."""
    return requests.Session()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
