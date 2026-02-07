"""LongMemEval benchmark harness.

Paper: https://arxiv.org/abs/2410.10813
Dataset: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
"""

import asyncio
import os
from pathlib import Path
from typing import Optional
from rich.console import Console

from ..shared.providers import Provider, TribalMemoryProvider
from ..shared.runner import run_benchmark
from ..shared.metrics import BenchmarkResult

console = Console()

DATASET_NAME = "xiaowu0162/longmemeval-cleaned"
CACHE_DIR = Path(__file__).parent / "data"


def download_dataset() -> dict:
    """Download LongMemEval dataset from HuggingFace."""
    from datasets import load_dataset
    
    console.print(f"[blue]Downloading {DATASET_NAME}...[/blue]")
    ds = load_dataset(DATASET_NAME, split="test")
    console.print(f"[green]Downloaded {len(ds)} examples[/green]")
    return ds


def parse_dataset(ds) -> tuple[list[dict], list[dict]]:
    """
    Parse LongMemEval dataset into conversations and questions.
    
    Returns:
        (conversations, questions) tuple
    """
    conversations = []
    questions = []
    
    for example in ds:
        # Extract conversation history
        history_sessions = example.get("history_sessions", [])
        
        # Each history session becomes a conversation
        for session in history_sessions:
            messages = []
            for turn in session:
                if isinstance(turn, dict):
                    messages.append({
                        "role": turn.get("role", "user"),
                        "content": turn.get("content", ""),
                    })
                elif isinstance(turn, list) and len(turn) >= 2:
                    messages.append({"role": "user", "content": turn[0]})
                    messages.append({"role": "assistant", "content": turn[1]})
            
            if messages:
                conversations.append({
                    "messages": messages,
                    "context": example.get("question_id", ""),
                })
        
        # Extract question
        questions.append({
            "id": example.get("question_id", ""),
            "question": example.get("question", ""),
            "expected": example.get("answer", ""),
            "category": example.get("question_type", "unknown"),
            "haystack_session": example.get("haystack_session_id", ""),
        })
    
    return conversations, questions


def answer_checker(expected: str, retrieved: list[str]) -> bool:
    """
    Check if the expected answer is contained in retrieved memories.
    
    Uses substring matching - the expected answer should appear
    somewhere in the retrieved content.
    """
    if not expected or not retrieved:
        return False
    
    expected_lower = expected.lower().strip()
    combined = " ".join(retrieved).lower()
    
    # Direct substring match
    if expected_lower in combined:
        return True
    
    # Check key phrases (split on common delimiters)
    key_phrases = [p.strip() for p in expected_lower.replace(",", "|").replace(";", "|").split("|")]
    for phrase in key_phrases:
        if phrase and phrase in combined:
            return True
    
    return False


async def run_longmemeval(
    provider: Optional[Provider] = None,
    sample: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> BenchmarkResult:
    """
    Run LongMemEval benchmark.
    
    Args:
        provider: Memory provider (defaults to TribalMemory)
        sample: Optional sample size (stratified by category)
        output_dir: Directory to save results
    
    Returns:
        BenchmarkResult with detailed metrics
    """
    if provider is None:
        provider = TribalMemoryProvider(instance="longmemeval")
    
    # Download and parse dataset
    ds = download_dataset()
    conversations, questions = parse_dataset(ds)
    
    console.print(f"\n[bold]Dataset parsed:[/bold]")
    console.print(f"  Conversations: {len(conversations)}")
    console.print(f"  Questions: {len(questions)}")
    
    # Run benchmark
    result = await run_benchmark(
        name="LongMemEval",
        provider=provider,
        conversations=conversations,
        questions=questions,
        answer_checker=answer_checker,
        sample=sample,
    )
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_path = output_dir / "longmemeval-results.json"
        json_path.write_text(result.to_json())
        console.print(f"\n[green]Results saved to {json_path}[/green]")
        
        # Save Markdown
        md_path = output_dir / "longmemeval-results.md"
        md_path.write_text(result.to_markdown())
        console.print(f"[green]Markdown saved to {md_path}[/green]")
    
    return result


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LongMemEval benchmark")
    parser.add_argument("--sample", type=int, help="Sample size (stratified)")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--provider-url", type=str, help="TribalMemory URL")
    parser.add_argument("--instance", type=str, default="longmemeval", help="Instance ID")
    
    args = parser.parse_args()
    
    provider = TribalMemoryProvider(
        base_url=args.provider_url,
        instance=args.instance,
    )
    
    result = asyncio.run(run_longmemeval(
        provider=provider,
        sample=args.sample,
        output_dir=Path(args.output),
    ))
    
    return 0 if result.overall_accuracy > 0.5 else 1


if __name__ == "__main__":
    exit(main())
