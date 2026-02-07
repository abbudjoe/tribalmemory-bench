"""Benchmark metrics and results."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json


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
            f"**Avg Latency:** {self.avg_latency_ms:.1f}ms",
            "",
            "## By Category",
            "",
            "| Category | Accuracy | Hit@1 | Hit@5 | Hit@10 | Latency |",
            "|----------|----------|-------|-------|--------|---------|",
        ]
        
        for cat in self.categories:
            lines.append(
                f"| {cat.category} | {cat.accuracy:.1%} ({cat.correct}/{cat.total}) | "
                f"{cat.hit_at_1:.1%} | {cat.hit_at_5:.1%} | {cat.hit_at_10:.1%} | "
                f"{cat.avg_latency_ms:.1f}ms |"
            )
        
        return "\n".join(lines)


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
        
        results.append(CategoryResult(
            category=category,
            total=total,
            correct=correct,
            accuracy=correct / total if total > 0 else 0,
            avg_latency_ms=avg_latency,
            hit_at_1=hit_1,
            hit_at_5=hit_5,
            hit_at_10=hit_10,
        ))
    
    return results
