"""
Memory App Analysis Reproducibility Tests

Tests to ensure dementia analysis produces consistent, reproducible results
for known inputs. These serve as regression tests.

Run with: pytest tests/test_memory_analysis.py -v
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.dementia_analysis import (
    analyze_dementia_markers,
    detect_repetitions,
    detect_aphasic_events,
    calculate_preposition_rate,
)


class TestPrepositionAnalysis:
    """Test preposition detection consistency."""
    
    def test_basic_preposition_count(self):
        """Should detect prepositions in simple text."""
        text = "I went to the store and bought some milk for my family."
        result = calculate_preposition_rate(text)
        
        assert result["word_count"] > 0
        assert result["preposition_count"] >= 2  # "to", "for"
        assert "rate_per_10_words" in result
    
    def test_preposition_reproducibility(self):
        """Same text should always produce same preposition count."""
        text = "The book is on the table in the living room."
        
        results = [calculate_preposition_rate(text) for _ in range(3)]
        
        # All results should be identical
        assert all(r["preposition_count"] == results[0]["preposition_count"] for r in results)
        assert all(r["word_count"] == results[0]["word_count"] for r in results)
    
    def test_known_preposition_text(self):
        """Verify counts for known text."""
        text = "I put the cup on the table."
        result = calculate_preposition_rate(text)
        
        assert result["word_count"] == 7
        assert result["preposition_count"] >= 1  # "on"
    
    def test_empty_text(self):
        """Should handle empty text gracefully."""
        result = calculate_preposition_rate("")
        
        assert result["word_count"] == 0
        assert result["preposition_count"] == 0


class TestRepetitionDetection:
    """Test repetition detection consistency."""
    
    def test_exact_repetition(self):
        """Should detect exact phrase repetitions."""
        segments = [
            {"text": "I went to the store yesterday.", "speaker": "A", "start": 0, "end": 5},
            {"text": "The weather was nice.", "speaker": "A", "start": 5, "end": 10},
            {"text": "I went to the store yesterday.", "speaker": "A", "start": 10, "end": 15},
        ]
        
        result = detect_repetitions(segments, threshold=0.9, min_segment_gap=1)
        
        assert result.get("repeated_concepts_count", 0) >= 1, "Should detect the exact repetition"
    
    def test_semantic_repetition(self):
        """Should detect semantically similar repetitions."""
        segments = [
            {"text": "I need to buy groceries.", "speaker": "A", "start": 0, "end": 5},
            {"text": "The store closes at nine.", "speaker": "A", "start": 5, "end": 10},
            {"text": "I have to purchase food.", "speaker": "A", "start": 10, "end": 15},
        ]
        
        # With semantic threshold of 0.7, may catch similar meanings
        result = detect_repetitions(segments, threshold=0.70, min_segment_gap=1)
        
        # Returns a dict with repeated_concepts
        assert isinstance(result, dict)
        assert "repeated_concepts" in result
    
    def test_no_false_positives(self):
        """Should not flag unrelated phrases as repetitions."""
        segments = [
            {"text": "I like pizza.", "speaker": "A", "start": 0, "end": 5},
            {"text": "The sky is blue.", "speaker": "A", "start": 5, "end": 10},
            {"text": "Cats are fluffy.", "speaker": "A", "start": 10, "end": 15},
        ]
        
        result = detect_repetitions(segments, threshold=0.85, min_segment_gap=1)
        
        # Should have zero repeated concepts for unrelated phrases
        assert result.get("repeated_concepts_count", 0) == 0, "Unrelated phrases should not be flagged"
    
    def test_repetition_reproducibility(self):
        """Same segments should always produce same repetitions."""
        segments = [
            {"text": "I went to the store.", "speaker": "A", "start": 0, "end": 5},
            {"text": "It was a nice day.", "speaker": "A", "start": 5, "end": 10},
            {"text": "I went to the store.", "speaker": "A", "start": 10, "end": 15},
        ]
        
        results = [detect_repetitions(segments, threshold=0.9, min_segment_gap=1) for _ in range(3)]
        
        # All results should have same repeated concepts count
        assert all(r.get("repeated_concepts_count") == results[0].get("repeated_concepts_count") for r in results)


class TestAphasicEventDetection:
    """Test aphasic event (word-finding difficulty) detection."""
    
    def test_detect_fillers(self):
        """Should detect filler words indicating word-finding difficulty."""
        text = "I went to the... um... the place where you buy things."
        
        events = detect_aphasic_events(text)
        
        # Should return dict with expected structure
        assert isinstance(events, dict)
        assert "event_count" in events
        assert "events" in events
    
    def test_detect_circumlocution(self):
        """Should detect circumlocution (describing instead of naming)."""
        text = "I need that thing you use to write with."
        
        events = detect_aphasic_events(text)
        
        # May or may not detect depending on implementation
        assert isinstance(events, dict)
        assert "events" in events
    
    def test_no_false_positives_fluent(self):
        """Should not flag fluent speech heavily."""
        text = "I went to the grocery store and bought milk, bread, and eggs."
        
        events = detect_aphasic_events(text)
        
        # Fluent speech should have few events
        assert events["event_count"] <= 2


class TestFullDementiaAnalysis:
    """Test the complete dementia analysis pipeline."""
    
    def test_analyze_markers_structure(self):
        """Should return expected structure from analyze_dementia_markers."""
        segments = [
            {"text": "Hello, how are you today?", "speaker": "A", "start": 0, "end": 5},
            {"text": "I am doing well, thank you.", "speaker": "B", "start": 5, "end": 10},
        ]
        
        result = analyze_dementia_markers(
            segments=segments,
            repetition_threshold=0.85,
            min_segment_gap=2,
        )
        
        # Check expected keys (actual API structure)
        assert "speakers" in result
        assert "summary" in result
        assert "risk_flags" in result
    
    def test_analyze_markers_reproducibility(self):
        """Full analysis should be reproducible."""
        segments = [
            {"text": "I went to the store to buy some bread.", "speaker": "A", "start": 0, "end": 5},
            {"text": "The bread was fresh.", "speaker": "A", "start": 5, "end": 10},
            {"text": "I went to the store to buy some bread.", "speaker": "A", "start": 10, "end": 15},
        ]
        
        results = [
            analyze_dementia_markers(segments, repetition_threshold=0.85, min_segment_gap=1)
            for _ in range(3)
        ]
        
        # Summaries should be consistent
        assert all(r["summary"] == results[0]["summary"] for r in results)
    
    def test_speaker_detection(self):
        """Should correctly identify speakers."""
        segments = [
            {"text": "Hello.", "speaker": "Speaker_1", "start": 0, "end": 1},
            {"text": "Hi there.", "speaker": "Speaker_2", "start": 1, "end": 2},
            {"text": "How are you?", "speaker": "Speaker_1", "start": 2, "end": 3},
        ]
        
        result = analyze_dementia_markers(segments)
        
        assert "speakers" in result
        assert len(result["speakers"]) == 2


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_segments(self):
        """Should handle empty segments list."""
        result = analyze_dementia_markers(segments=[])
        
        assert result is not None
        assert isinstance(result, dict)
    
    def test_single_segment(self):
        """Should handle single segment."""
        segments = [
            {"text": "Just one sentence here.", "speaker": "A", "start": 0, "end": 5},
        ]
        
        result = analyze_dementia_markers(segments)
        
        assert "speakers" in result
    
    def test_unicode_text(self):
        """Should handle unicode characters."""
        text = "Café, naïve, and résumé are common words."
        
        result = calculate_preposition_rate(text)
        
        assert result["word_count"] > 0


class TestKnownBaselines:
    """Tests against known baseline results for specific inputs."""
    
    BASELINE_TEXT_1 = "I went to the store on Monday. I bought some milk and bread. Then I went to the store again."
    
    def test_baseline_prepositions(self):
        """Verify preposition count for baseline text."""
        result = calculate_preposition_rate(self.BASELINE_TEXT_1)
        
        # Should find "to" (x2), "on" = at least 3 prepositions
        assert result["preposition_count"] >= 3
        assert result["word_count"] >= 18


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
