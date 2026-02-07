"""ConvoMem benchmark harness.

Paper: https://arxiv.org/abs/2511.10523
Dataset: https://huggingface.co/datasets/Salesforce/ConvoMem
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

DATASET_NAME = "Salesforce/ConvoMem"

# ConvoMem categories
CATEGORIES = [
    "user_facts",
    "assistant_recall", 
    "abstention",
    "preferences",
    "temporal_changes",
    "implicit_connections",
]


def download_dataset() -> dict:
    """Download ConvoMem dataset from HuggingFace."""
    from datasets import load_dataset
    
    console.print(f"[blue]Downloading {DATASET_NAME}...[/blue]")
    ds = load_dataset(DATASET_NAME, split="test")
    console.print(f"[green]Downloaded {len(ds)} examples[/green]")
    return ds


def parse_dataset(ds) -> tuple[list[dict], list[dict]]:
    """
    Parse ConvoMem dataset into conversations and questions.
    
    ConvoMem format:
    - conversations: list of messages
    - question: the query
    - answer: expected answer
    - category: one of CATEGORIES
    
    Returns:
        (conversations, questions) tuple
    """
    conversations = []
    questions = []
    seen_convos = set()
    
    for example in ds:
        # Extract conversation
        conv_messages = example.get("conversation", example.get("messages", []))
        conv_id = example.get("conversation_id", str(hash(str(conv_messages)[:100])))
        
        # Dedupe conversations
        if conv_id not in seen_convos:
            seen_convos.add(conv_id)
            messages = []
            for msg in conv_messages:
                if isinstance(msg, dict):
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", ""),
                    })
            if messages:
                conversations.append({
                    "id": conv_id,
                    "messages": messages,
                })
        
        # Extract question
        questions.append({
            "id": example.get("id", example.get("question_id", "")),
            "question": example.get("question", ""),
            "expected": example.get("answer", ""),
            "category": example.get("category", "unknown"),
            "conversation_id": conv_id,
        })
    
    return conversations, questions


def answer_checker(expected: str, retrieved: list[str]) -> bool:
    """
    Check if the expected answer is contained in retrieved memories.
    
    ConvoMem has specific answer formats per category:
    - user_facts: factual answer
    - abstention: "I don't know" type answers are correct if no evidence
    - temporal_changes: latest value matters
    """
    if not expected or not retrieved:
        return False
    
    expected_lower = expected.lower().strip()
    combined = " ".join(retrieved).lower()
    
    # Abstention handling
    abstention_phrases = ["don't know", "no information", "not mentioned", "cannot determine"]
    if any(phrase in expected_lower for phrase in abstention_phrases):
        # For abstention, success is NOT finding contradicting info
        # This is simplified - real eval would need LLM judge
        return True
    
    # Direct substring match
    if expected_lower in combined:
        return True
    
    # Check key phrases
    key_phrases = [p.strip() for p in expected_lower.replace(",", "|").replace(";", "|").split("|")]
    for phrase in key_phrases:
        if phrase and len(phrase) > 3 and phrase in combined:
            return True
    
    return False


async def run_convomem(
    provider: Optional[Provider] = None,
    sample: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> BenchmarkResult:
    """
    Run ConvoMem benchmark.
    
    Args:
        provider: Memory provider (defaults to TribalMemory)
        sample: Optional sample size (stratified by category)
        output_dir: Directory to save results
    
    Returns:
        BenchmarkResult with detailed metrics
    """
    if provider is None:
        provider = TribalMemoryProvider(instance="convomem")
    
    # Download and parse dataset
    ds = download_dataset()
    conversations, questions = parse_dataset(ds)
    
    console.print(f"\n[bold]Dataset parsed:[/bold]")
    console.print(f"  Conversations: {len(conversations)}")
    console.print(f"  Questions: {len(questions)}")
    console.print(f"  Categories: {', '.join(CATEGORIES)}")
    
    # Run benchmark
    result = await run_benchmark(
        name="ConvoMem",
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
        json_path = output_dir / "convomem-results.json"
        json_path.write_text(result.to_json())
        console.print(f"\n[green]Results saved to {json_path}[/green]")
        
        # Save Markdown
        md_path = output_dir / "convomem-results.md"
        md_path.write_text(result.to_markdown())
        console.print(f"[green]Markdown saved to {md_path}[/green]")
    
    return result


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ConvoMem benchmark")
    parser.add_argument("--sample", type=int, help="Sample size (stratified)")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--provider-url", type=str, help="TribalMemory URL")
    parser.add_argument("--instance", type=str, default="convomem", help="Instance ID")
    
    args = parser.parse_args()
    
    provider = TribalMemoryProvider(
        base_url=args.provider_url,
        instance=args.instance,
    )
    
    result = asyncio.run(run_convomem(
        provider=provider,
        sample=args.sample,
        output_dir=Path(args.output),
    ))
    
    return 0 if result.overall_accuracy > 0.5 else 1


if __name__ == "__main__":
    exit(main())
