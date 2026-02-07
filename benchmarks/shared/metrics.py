"""Benchmark metrics and results."""

from dataclasses import dataclass, field
from typing import Optional
import json
import statistics


@dataclass
class QuestionResult:
    """Result for a single question."""
    question_id: str
    category: str
    question: str
    expected: str
    retrieved: list[str]
    correct: bool
    latency_ms: float
    hit_at_k: dict[int, bool] = field(default_factory=dict)  # Hit@1, Hit@5, Hit@10
    reciprocal_rank: float = 0.0  # 1/rank of first correct, 0 if not found


@dataclass
class CategoryResult:
    """Results for a category of questions."""
    category: str
    total: int
    correct: int
    accuracy: float
    avg_latency_ms: float
    hit_at_1: float = 0.0
    hit_at_5: float = 0.0
    hit_at_10: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""
    benchmark: str
    provider: str
    timestamp: str
    total_questions: int
    total_correct: int
    overall_accuracy: float
    avg_latency_ms: float
    mrr: float  # Overall Mean Reciprocal Rank
    categories: list[CategoryResult]
    questions: list[QuestionResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "benchmark": self.benchmark,
            "provider": self.provider,
            "timestamp": self.timestamp,
            "total_questions": self.total_questions,
            "total_correct": self.total_correct,
            "overall_accuracy": self.overall_accuracy,
            "avg_latency_ms": self.avg_latency_ms,
            "mrr": self.mrr,
            "categories": [
                {
                    "category": c.category,
                    "total": c.total,
                    "correct": c.correct,
                    "accuracy": c.accuracy,
                    "avg_latency_ms": c.avg_latency_ms,
                    "hit_at_1": c.hit_at_1,
                    "hit_at_5": c.hit_at_5,
                    "hit_at_10": c.hit_at_10,
                    "mrr": c.mrr,
                }
                for c in self.categories
            ],
            "metadata": self.metadata,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# {self.benchmark} Results",
            "",
            f"**Provider:** {self.provider}",
            f"**Timestamp:** {self.timestamp}",
            f"**Overall Accuracy:** {self.overall_accuracy:.1%} ({self.total_correct}/{self.total_questions})",
            f"**MRR:** {self.mrr:.3f}",
            f"**Avg Latency:** {self.avg_latency_ms:.1f}ms",
            "",
            "## By Category",
            "",
            "| Category | Accuracy | MRR | Hit@1 | Hit@5 | Hit@10 | Latency |",
            "|----------|----------|-----|-------|-------|--------|---------|",
        ]
        
        for cat in self.categories:
            lines.append(
                f"| {cat.category} | {cat.accuracy:.1%} ({cat.correct}/{cat.total}) | "
                f"{cat.mrr:.3f} | {cat.hit_at_1:.1%} | {cat.hit_at_5:.1%} | {cat.hit_at_10:.1%} | "
                f"{cat.avg_latency_ms:.1f}ms |"
            )
        
        # Add metadata
        if self.metadata:
            lines.extend([
                "",
                "## Metadata",
                "",
            ])
            for key, value in self.metadata.items():
                lines.append(f"- **{key}:** {value}")
        
        return "\n".join(lines)
    
    def confidence_interval(self, confidence: float = 0.95) -> tuple[float, float]:
        """
        Compute confidence interval for accuracy.
        
        Uses normal approximation for proportion.
        """
        import math
        
        n = self.total_questions
        p = self.overall_accuracy
        
        if n == 0:
            return (0.0, 0.0)
        
        # Z-score for confidence level
        z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
        
        # Standard error
        se = math.sqrt(p * (1 - p) / n)
        
        lower = max(0.0, p - z * se)
        upper = min(1.0, p + z * se)
        
        return (lower, upper)


def compute_category_results(questions: list[QuestionResult]) -> list[CategoryResult]:
    """Compute per-category results from question results."""
    from collections import defaultdict
    
    by_category: dict[str, list[QuestionResult]] = defaultdict(list)
    for q in questions:
        by_category[q.category].append(q)
    
    results = []
    for category, qs in sorted(by_category.items()):
        correct = sum(1 for q in qs if q.correct)
        total = len(qs)
        avg_latency = sum(q.latency_ms for q in qs) / total if total > 0 else 0
        
        # Compute Hit@K
        hit_1 = sum(1 for q in qs if q.hit_at_k.get(1, False)) / total if total > 0 else 0
        hit_5 = sum(1 for q in qs if q.hit_at_k.get(5, False)) / total if total > 0 else 0
        hit_10 = sum(1 for q in qs if q.hit_at_k.get(10, False)) / total if total > 0 else 0
        
        # Compute MRR
        mrr = sum(q.reciprocal_rank for q in qs) / total if total > 0 else 0
        
        results.append(CategoryResult(
            category=category,
            total=total,
            correct=correct,
            accuracy=correct / total if total > 0 else 0,
            avg_latency_ms=avg_latency,
            hit_at_1=hit_1,
            hit_at_5=hit_5,
            hit_at_10=hit_10,
            mrr=mrr,
        ))
    
    return results


def compute_reciprocal_rank(
    expected: str,
    retrieved: list[str],
    checker_fn,
) -> float:
    """
    Compute reciprocal rank for a single question.
    
    Returns 1/rank of first correct result, or 0 if not found.
    """
    for i, content in enumerate(retrieved, 1):
        if checker_fn(expected, [content]):
            return 1.0 / i
    return 0.0


def compare_results(
    result_a: BenchmarkResult,
    result_b: BenchmarkResult,
) -> dict:
    """
    Compare two benchmark results.
    
    Returns dict with comparison metrics.
    """
    return {
        "provider_a": result_a.provider,
        "provider_b": result_b.provider,
        "accuracy_diff": result_a.overall_accuracy - result_b.overall_accuracy,
        "mrr_diff": result_a.mrr - result_b.mrr,
        "latency_diff_ms": result_a.avg_latency_ms - result_b.avg_latency_ms,
        "winner_accuracy": result_a.provider if result_a.overall_accuracy > result_b.overall_accuracy else result_b.provider,
        "winner_mrr": result_a.provider if result_a.mrr > result_b.mrr else result_b.provider,
        "winner_latency": result_a.provider if result_a.avg_latency_ms < result_b.avg_latency_ms else result_b.provider,
    }
