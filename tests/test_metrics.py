"""Tests for metrics module."""

import pytest
from benchmarks.shared.metrics import (
    QuestionResult,
    CategoryResult,
    BenchmarkResult,
    compute_category_results,
    compute_reciprocal_rank,
    compare_results,
)


class TestComputeReciprocalRank:
    def test_first_position_match(self):
        """RR = 1 when answer is in first position."""
        def checker(expected, retrieved):
            return expected.lower() in " ".join(retrieved).lower()
        
        rr = compute_reciprocal_rank("hello", ["hello world", "goodbye"], checker)
        assert rr == 1.0
    
    def test_second_position_match(self):
        """RR = 0.5 when answer is in second position."""
        def checker(expected, retrieved):
            return expected.lower() in " ".join(retrieved).lower()
        
        rr = compute_reciprocal_rank("world", ["hello", "world here"], checker)
        assert rr == 0.5
    
    def test_third_position_match(self):
        """RR = 0.333... when answer is in third position."""
        def checker(expected, retrieved):
            return expected.lower() in " ".join(retrieved).lower()
        
        rr = compute_reciprocal_rank("third", ["a", "b", "third"], checker)
        assert abs(rr - 1/3) < 0.001
    
    def test_no_match(self):
        """RR = 0 when answer not found."""
        def checker(expected, retrieved):
            return expected.lower() in " ".join(retrieved).lower()
        
        rr = compute_reciprocal_rank("missing", ["a", "b", "c"], checker)
        assert rr == 0.0
    
    def test_empty_retrieved(self):
        """RR = 0 with empty retrieved list."""
        def checker(expected, retrieved):
            return False
        
        rr = compute_reciprocal_rank("test", [], checker)
        assert rr == 0.0


class TestComputeCategoryResults:
    def test_single_category(self):
        questions = [
            QuestionResult(
                question_id="1", category="A", question="q1",
                expected="a1", retrieved=[], correct=True,
                latency_ms=10.0, hit_at_k={1: True, 5: True, 10: True},
                reciprocal_rank=1.0
            ),
            QuestionResult(
                question_id="2", category="A", question="q2",
                expected="a2", retrieved=[], correct=False,
                latency_ms=20.0, hit_at_k={1: False, 5: True, 10: True},
                reciprocal_rank=0.5
            ),
        ]
        
        results = compute_category_results(questions)
        
        assert len(results) == 1
        assert results[0].category == "A"
        assert results[0].total == 2
        assert results[0].correct == 1
        assert results[0].accuracy == 0.5
        assert results[0].avg_latency_ms == 15.0
        assert results[0].hit_at_1 == 0.5
        assert results[0].mrr == 0.75  # (1.0 + 0.5) / 2
    
    def test_multiple_categories(self):
        questions = [
            QuestionResult(
                question_id="1", category="A", question="q1",
                expected="a1", retrieved=[], correct=True,
                latency_ms=10.0, reciprocal_rank=1.0
            ),
            QuestionResult(
                question_id="2", category="B", question="q2",
                expected="a2", retrieved=[], correct=True,
                latency_ms=20.0, reciprocal_rank=1.0
            ),
        ]
        
        results = compute_category_results(questions)
        
        assert len(results) == 2
        categories = {r.category for r in results}
        assert categories == {"A", "B"}
    
    def test_empty_questions(self):
        results = compute_category_results([])
        assert results == []


class TestBenchmarkResult:
    def test_to_dict(self):
        result = BenchmarkResult(
            benchmark="TestBench",
            provider="TestProvider",
            timestamp="2026-02-07T00:00:00",
            total_questions=10,
            total_correct=8,
            overall_accuracy=0.8,
            avg_latency_ms=50.0,
            mrr=0.9,
            categories=[],
            metadata={"key": "value"}
        )
        
        d = result.to_dict()
        
        assert d["benchmark"] == "TestBench"
        assert d["overall_accuracy"] == 0.8
        assert d["mrr"] == 0.9
        assert d["metadata"]["key"] == "value"
    
    def test_to_json(self):
        result = BenchmarkResult(
            benchmark="Test",
            provider="Provider",
            timestamp="2026-02-07T00:00:00",
            total_questions=1,
            total_correct=1,
            overall_accuracy=1.0,
            avg_latency_ms=10.0,
            mrr=1.0,
            categories=[],
        )
        
        json_str = result.to_json()
        
        assert '"benchmark": "Test"' in json_str
        assert '"mrr": 1.0' in json_str
    
    def test_to_markdown(self):
        result = BenchmarkResult(
            benchmark="Test",
            provider="Provider",
            timestamp="2026-02-07T00:00:00",
            total_questions=10,
            total_correct=8,
            overall_accuracy=0.8,
            avg_latency_ms=50.0,
            mrr=0.9,
            categories=[
                CategoryResult(
                    category="cat1",
                    total=10,
                    correct=8,
                    accuracy=0.8,
                    avg_latency_ms=50.0,
                    hit_at_1=0.7,
                    hit_at_5=0.8,
                    hit_at_10=0.9,
                    mrr=0.85,
                )
            ],
        )
        
        md = result.to_markdown()
        
        assert "# Test Results" in md
        assert "**MRR:** 0.900" in md
        assert "| cat1 |" in md
    
    def test_confidence_interval(self):
        result = BenchmarkResult(
            benchmark="Test",
            provider="Provider",
            timestamp="2026-02-07T00:00:00",
            total_questions=100,
            total_correct=80,
            overall_accuracy=0.8,
            avg_latency_ms=50.0,
            mrr=0.9,
            categories=[],
        )
        
        lower, upper = result.confidence_interval(0.95)
        
        # 80% accuracy with n=100 should have CI roughly [0.72, 0.88]
        assert 0.70 < lower < 0.80
        assert 0.80 < upper < 0.90


class TestCompareResults:
    def test_compare_accuracy(self):
        result_a = BenchmarkResult(
            benchmark="Test",
            provider="ProviderA",
            timestamp="2026-02-07T00:00:00",
            total_questions=10,
            total_correct=8,
            overall_accuracy=0.8,
            avg_latency_ms=50.0,
            mrr=0.9,
            categories=[],
        )
        result_b = BenchmarkResult(
            benchmark="Test",
            provider="ProviderB",
            timestamp="2026-02-07T00:00:00",
            total_questions=10,
            total_correct=6,
            overall_accuracy=0.6,
            avg_latency_ms=40.0,
            mrr=0.7,
            categories=[],
        )
        
        comparison = compare_results(result_a, result_b)
        
        assert abs(comparison["accuracy_diff"] - 0.2) < 0.001
        assert comparison["winner_accuracy"] == "ProviderA"
        assert comparison["winner_latency"] == "ProviderB"  # Lower is better
        assert comparison["winner_mrr"] == "ProviderA"
