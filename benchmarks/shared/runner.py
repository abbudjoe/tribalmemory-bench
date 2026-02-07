"""Benchmark runner orchestration."""

import asyncio
import random
import time
from datetime import datetime
from typing import Callable, Optional
from rich.console import Console
from rich.progress import Progress

from .providers import Provider
from .metrics import (
    BenchmarkResult,
    QuestionResult,
    compute_category_results,
    compute_reciprocal_rank,
)

console = Console()

# Default batch size for ingestion
DEFAULT_BATCH_SIZE = 20
# Default concurrency for queries
DEFAULT_QUERY_CONCURRENCY = 10


async def run_benchmark(
    name: str,
    provider: Provider,
    conversations: list[dict],
    questions: list[dict],
    answer_checker: Callable[[str, list[str]], bool],
    sample: Optional[int] = None,
    seed: int = 42,
    batch_size: int = DEFAULT_BATCH_SIZE,
    query_concurrency: int = DEFAULT_QUERY_CONCURRENCY,
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
        seed: Random seed for reproducibility
        batch_size: Number of memories to ingest per batch
        query_concurrency: Max concurrent queries
        show_progress: Show progress bar
    
    Returns:
        BenchmarkResult with detailed metrics
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Sample questions if requested
    if sample and sample < len(questions):
        questions = stratified_sample(questions, sample, seed)
    
    console.print(f"[bold blue]Running {name}[/bold blue]")
    console.print(f"  Conversations: {len(conversations)}")
    console.print(f"  Questions: {len(questions)}")
    console.print(f"  Seed: {seed}")
    
    # Phase 1: Ingest conversations
    console.print("\n[bold]Phase 1: Ingesting conversations...[/bold]")
    ingest_start = time.time()
    
    # Chunk conversations into memory units
    memory_chunks = chunk_conversations(conversations)
    console.print(f"  Memory chunks: {len(memory_chunks)}")
    
    # Batch ingestion
    with Progress() as progress:
        task = progress.add_task("Ingesting", total=len(memory_chunks))
        
        for i in range(0, len(memory_chunks), batch_size):
            batch = memory_chunks[i:i + batch_size]
            try:
                await provider.store_batch(batch)
            except Exception as e:
                # Log batch failure and fall back to sequential
                console.print(
                    f"[yellow]Batch ingestion failed ({e}), "
                    f"falling back to sequential[/yellow]"
                )
                for chunk in batch:
                    await provider.store(chunk["content"], chunk.get("context"))
            progress.advance(task, len(batch))
    
    ingest_time = time.time() - ingest_start
    console.print(f"  Ingestion completed in {ingest_time:.1f}s")
    
    # Phase 2: Run questions concurrently
    console.print("\n[bold]Phase 2: Running questions...[/bold]")
    query_start = time.time()
    
    results: list[QuestionResult] = []
    semaphore = asyncio.Semaphore(query_concurrency)
    
    async def run_with_semaphore(q: dict) -> QuestionResult:
        async with semaphore:
            return await run_question(provider, q, answer_checker)
    
    with Progress() as progress:
        task = progress.add_task("Querying", total=len(questions))
        
        # Run queries concurrently with semaphore
        tasks = [run_with_semaphore(q) for q in questions]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            progress.advance(task)
    
    query_time = time.time() - query_start
    console.print(f"  Queries completed in {query_time:.1f}s")
    
    # Compute results
    category_results = compute_category_results(results)
    total_correct = sum(1 for r in results if r.correct)
    avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0
    overall_mrr = sum(r.reciprocal_rank for r in results) / len(results) if results else 0
    
    benchmark_result = BenchmarkResult(
        benchmark=name,
        provider=provider.__class__.__name__,
        timestamp=datetime.utcnow().isoformat(),
        total_questions=len(results),
        total_correct=total_correct,
        overall_accuracy=total_correct / len(results) if results else 0,
        avg_latency_ms=avg_latency,
        mrr=overall_mrr,
        categories=category_results,
        questions=results,
        metadata={
            "sample_size": sample,
            "seed": seed,
            "ingest_time_s": ingest_time,
            "query_time_s": query_time,
            "conversation_count": len(conversations),
            "memory_chunk_count": len(memory_chunks),
            "batch_size": batch_size,
            "query_concurrency": query_concurrency,
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


def chunk_conversations(conversations: list[dict]) -> list[dict]:
    """
    Convert conversations into memory chunks.
    
    Strategy: Combine user-assistant turn pairs into chunks.
    This preserves conversational context while creating meaningful units.
    """
    chunks = []
    
    for conv in conversations:
        messages = conv.get("messages", [])
        context = conv.get("context", "")
        session_id = conv.get("session", conv.get("id", ""))
        
        # Group messages into turn pairs (user + assistant response)
        current_chunk = []
        for msg in messages:
            current_chunk.append(msg)
            
            # Complete a chunk when we have a user-assistant pair
            # or hit 4 messages (2 turns)
            if len(current_chunk) >= 4 or (
                len(current_chunk) >= 2 and 
                current_chunk[-1].get("role") == "assistant"
            ):
                chunk_content = "\n".join(
                    f"{m.get('role', 'user')}: {m.get('content', '')}"
                    for m in current_chunk
                )
                chunks.append({
                    "content": chunk_content,
                    "context": f"session:{session_id}" if session_id else context,
                })
                current_chunk = []
        
        # Don't forget remaining messages
        if current_chunk:
            chunk_content = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}"
                for m in current_chunk
            )
            chunks.append({
                "content": chunk_content,
                "context": f"session:{session_id}" if session_id else context,
            })
    
    return chunks


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
    
    # Compute reciprocal rank
    rr = compute_reciprocal_rank(expected, retrieved, answer_checker)
    
    return QuestionResult(
        question_id=question_id,
        category=category,
        question=query,
        expected=expected,
        retrieved=retrieved,
        correct=correct,
        latency_ms=latency_ms,
        hit_at_k=hit_at_k,
        reciprocal_rank=rr,
    )


def stratified_sample(questions: list[dict], n: int, seed: int = 42) -> list[dict]:
    """
    Sample n questions, stratified by category.
    
    Args:
        questions: List of question dicts with 'category' key
        n: Number of questions to sample
        seed: Random seed for reproducibility
    
    Returns:
        Stratified sample of questions
    """
    from collections import defaultdict
    
    # Use a separate random instance for reproducibility
    rng = random.Random(seed)
    
    by_category: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        by_category[q.get("category", "unknown")].append(q)
    
    # Proportional allocation
    total = len(questions)
    sampled = []
    
    for category, qs in sorted(by_category.items()):
        # Proportional sample from each category
        cat_n = max(1, int(n * len(qs) / total))
        sampled.extend(rng.sample(qs, min(cat_n, len(qs))))
    
    # Adjust to exact n if needed
    if len(sampled) > n:
        sampled = rng.sample(sampled, n)
    elif len(sampled) < n:
        # Add more from largest categories
        remaining = [q for q in questions if q not in sampled]
        if remaining:
            sampled.extend(rng.sample(remaining, min(n - len(sampled), len(remaining))))
    
    # Shuffle to avoid category clustering
    rng.shuffle(sampled)
    
    return sampled
