"""Benchmark runner orchestration."""

import asyncio
import time
from datetime import datetime
from typing import Callable, Optional, Any
from rich.console import Console
from rich.progress import Progress, TaskID

from .providers import Provider
from .metrics import (
    BenchmarkResult,
    QuestionResult,
    compute_category_results,
)

console = Console()


async def run_benchmark(
    name: str,
    provider: Provider,
    conversations: list[dict],
    questions: list[dict],
    answer_checker: Callable[[str, list[str]], bool],
    sample: Optional[int] = None,
    show_progress: bool = True,
) -> BenchmarkResult:
    """
    Run a benchmark against a provider.
    
    Args:
        name: Benchmark name
        provider: Memory provider to test
        conversations: List of conversation dicts to ingest
        questions: List of question dicts with 'question', 'expected', 'category'
        answer_checker: Function to check if retrieved memories contain the answer
        sample: Optional sample size (stratified by category)
        show_progress: Show progress bar
    
    Returns:
        BenchmarkResult with detailed metrics
    """
    # Sample questions if requested
    if sample and sample < len(questions):
        questions = stratified_sample(questions, sample)
    
    console.print(f"[bold blue]Running {name}[/bold blue]")
    console.print(f"  Conversations: {len(conversations)}")
    console.print(f"  Questions: {len(questions)}")
    
    # Phase 1: Ingest conversations
    console.print("\n[bold]Phase 1: Ingesting conversations...[/bold]")
    ingest_start = time.time()
    
    with Progress() as progress:
        task = progress.add_task("Ingesting", total=len(conversations))
        for conv in conversations:
            await ingest_conversation(provider, conv)
            progress.advance(task)
    
    ingest_time = time.time() - ingest_start
    console.print(f"  Ingestion completed in {ingest_time:.1f}s")
    
    # Phase 2: Run questions
    console.print("\n[bold]Phase 2: Running questions...[/bold]")
    results: list[QuestionResult] = []
    
    with Progress() as progress:
        task = progress.add_task("Querying", total=len(questions))
        
        for q in questions:
            result = await run_question(provider, q, answer_checker)
            results.append(result)
            progress.advance(task)
    
    # Compute results
    category_results = compute_category_results(results)
    total_correct = sum(1 for r in results if r.correct)
    avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0
    
    benchmark_result = BenchmarkResult(
        benchmark=name,
        provider=provider.__class__.__name__,
        timestamp=datetime.utcnow().isoformat(),
        total_questions=len(results),
        total_correct=total_correct,
        overall_accuracy=total_correct / len(results) if results else 0,
        avg_latency_ms=avg_latency,
        categories=category_results,
        questions=results,
        metadata={
            "sample_size": sample,
            "ingest_time_s": ingest_time,
            "conversation_count": len(conversations),
        }
    )
    
    # Print summary
    console.print("\n[bold green]Results:[/bold green]")
    console.print(f"  Overall: {benchmark_result.overall_accuracy:.1%} ({total_correct}/{len(results)})")
    console.print(f"  Avg Latency: {avg_latency:.1f}ms")
    console.print("\n  By Category:")
    for cat in category_results:
        console.print(f"    {cat.category}: {cat.accuracy:.1%} ({cat.correct}/{cat.total})")
    
    return benchmark_result


async def ingest_conversation(provider: Provider, conversation: dict) -> None:
    """Ingest a conversation into the provider."""
    messages = conversation.get("messages", [])
    context = conversation.get("context", "")
    
    # Combine messages into memory chunks
    # Strategy: store each message pair or chunk
    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        if content:
            await provider.store(content, context=context)


async def run_question(
    provider: Provider,
    question: dict,
    answer_checker: Callable[[str, list[str]], bool],
) -> QuestionResult:
    """Run a single question and check the answer."""
    query = question["question"]
    expected = question["expected"]
    category = question.get("category", "unknown")
    question_id = question.get("id", "")
    
    # Time the recall
    start = time.time()
    memories = await provider.recall(query, limit=10)
    latency_ms = (time.time() - start) * 1000
    
    # Extract content from memories
    retrieved = [m.content for m in memories]
    
    # Check correctness
    correct = answer_checker(expected, retrieved)
    
    # Compute Hit@K
    hit_at_k = {}
    for k in [1, 5, 10]:
        hit_at_k[k] = answer_checker(expected, retrieved[:k])
    
    return QuestionResult(
        question_id=question_id,
        category=category,
        question=query,
        expected=expected,
        retrieved=retrieved,
        correct=correct,
        latency_ms=latency_ms,
        hit_at_k=hit_at_k,
    )


def stratified_sample(questions: list[dict], n: int) -> list[dict]:
    """Sample n questions, stratified by category."""
    from collections import defaultdict
    import random
    
    by_category: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        by_category[q.get("category", "unknown")].append(q)
    
    # Proportional allocation
    total = len(questions)
    sampled = []
    
    for category, qs in by_category.items():
        # Proportional sample from each category
        cat_n = max(1, int(n * len(qs) / total))
        sampled.extend(random.sample(qs, min(cat_n, len(qs))))
    
    # Adjust to exact n if needed
    if len(sampled) > n:
        sampled = random.sample(sampled, n)
    elif len(sampled) < n:
        # Add more from largest categories
        remaining = [q for q in questions if q not in sampled]
        sampled.extend(random.sample(remaining, min(n - len(sampled), len(remaining))))
    
    return sampled
