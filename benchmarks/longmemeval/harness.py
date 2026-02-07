"""LongMemEval benchmark harness.

Paper: https://arxiv.org/abs/2410.10813
Dataset: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned

Why direct JSON download instead of HuggingFace datasets library?
- The datasets library has pyarrow type coercion issues with the mixed-type
  'answer' column (some answers are strings, some are integers).
- Streaming mode was attempted but also fails on the same type coercion.
- Direct JSON download is reliable and simple, with local file caching.
"""

import json
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

# Direct HuggingFace URLs for JSON download
_HF_BASE_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"
)
DATASET_URLS = {
    "s": f"{_HF_BASE_URL}/longmemeval_s_cleaned.json",
    "m": f"{_HF_BASE_URL}/longmemeval_m_cleaned.json",
}
CACHE_DIR = Path(__file__).parent / "data"


def download_dataset(variant: str = "s") -> list[dict]:
    """
    Download LongMemEval dataset directly from HuggingFace.

    Uses direct JSON download to avoid pyarrow type coercion issues with
    the mixed-type 'answer' column in the datasets library.

    Args:
        variant: Dataset variant - "s" (small, 500) or "m" (medium, 2000+)

    Returns:
        List of question examples
    """
    import httpx

    if variant not in DATASET_URLS:
        raise ValueError(f"Unknown variant '{variant}'. Use 's' or 'm'.")

    url = DATASET_URLS[variant]
    cache_path = CACHE_DIR / f"longmemeval_{variant}.json"

    # Use cached version if available
    if cache_path.exists():
        console.print(f"[blue]Loading cached dataset ({variant})...[/blue]")
        return json.loads(cache_path.read_text())

    console.print(f"[blue]Downloading {DATASET_NAME} ({variant})...[/blue]")

    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        data = response.json()

    # Cache for future runs
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data))

    console.print(f"[green]Downloaded {len(data)} examples (cached)[/green]")
    return data


def parse_dataset(ds: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Parse LongMemEval dataset into conversations and questions.

    Returns:
        (conversations, questions) tuple where conversations are deduplicated
        across the entire dataset (sessions repeat across questions).
    """
    # Parse all sessions without filtering
    return _parse_dataset_filtered(ds, needed_sessions=None)


def _parse_dataset_filtered(
    ds: list[dict],
    needed_sessions: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Parse LongMemEval dataset, optionally filtering to needed sessions.

    This is an internal function used by parse_dataset() and run_longmemeval().
    It's private because the filtering logic is an implementation detail of
    the sampling optimization.

    Args:
        ds: Raw dataset examples
        needed_sessions: Set of session IDs to include (None = include all)

    Returns:
        (conversations, questions) tuple
    """
    seen_sessions: set[str] = set()  # Track to avoid duplicates
    conversations = []
    questions = []
    missing_id_count = 0  # Track synthetic ID usage for warning

    for example in ds:
        # Extract conversation history (haystack_sessions in this dataset)
        haystack_sessions = example.get("haystack_sessions", [])
        session_ids = example.get("haystack_session_ids", [])

        # Each haystack session becomes a conversation
        for i, session in enumerate(haystack_sessions):
            # Create session ID for deduplication
            if i < len(session_ids):
                session_id = session_ids[i]
            else:
                session_id = f"session_{i}"
                missing_id_count += 1

            # Skip if filtering and not needed
            if needed_sessions is not None and session_id not in needed_sessions:
                continue

            # Skip if we've already seen this session
            if session_id in seen_sessions:
                continue
            seen_sessions.add(session_id)

            messages = []
            for turn in session:
                if isinstance(turn, dict):
                    messages.append({
                        "role": turn.get("role", "user"),
                        "content": turn.get("content", ""),
                    })
                elif isinstance(turn, list) and len(turn) >= 2:
                    # Legacy format: [user_msg, assistant_msg]
                    messages.append({"role": "user", "content": turn[0]})
                    messages.append({"role": "assistant", "content": turn[1]})

            if messages:
                conversations.append({
                    "id": session_id,
                    "messages": messages,
                    "context": example.get("question_id", ""),
                })

        # Extract question
        questions.append({
            "id": example.get("question_id", ""),
            "question": example.get("question", ""),
            "expected": example.get("answer", ""),
            "category": example.get("question_type", "unknown"),
            "answer_session_ids": example.get("answer_session_ids", []),
        })

    # Warn if we had to use synthetic session IDs (indicates dataset issue)
    if missing_id_count > 0:
        console.print(
            f"[yellow]Warning: {missing_id_count} sessions missing IDs, "
            f"used synthetic IDs[/yellow]"
        )

    return conversations, questions


def answer_checker(expected: str | int, retrieved: list[str]) -> bool:
    """
    Check if the expected answer is contained in retrieved memories.

    Uses substring matching - the expected answer should appear
    somewhere in the retrieved content.
    """
    if expected is None or expected == "" or not retrieved:
        return False

    # Handle int answers (some questions have numeric answers)
    expected_lower = str(expected).lower().strip()
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
    variant: str = "s",
    seed: int = 42,
) -> BenchmarkResult:
    """
    Run LongMemEval benchmark.

    Args:
        provider: Memory provider (defaults to TribalMemory)
        sample: Optional sample size (stratified by category)
        output_dir: Directory to save results
        variant: Dataset variant - "s" (500 questions) or "m" (2000+)
        seed: Random seed for reproducibility

    Returns:
        BenchmarkResult with detailed metrics
    """
    from ..shared.runner import stratified_sample

    if provider is None:
        provider = TribalMemoryProvider(instance="longmemeval")

    # Download raw dataset
    ds = download_dataset(variant=variant)

    console.print(f"\n[bold]Dataset loaded:[/bold]")
    console.print(f"  Total questions: {len(ds)}")

    # Sample questions FIRST from raw dataset (before parsing)
    raw_questions = ds
    if sample and sample < len(ds):
        # Convert to format stratified_sample expects
        temp_questions = [
            {"raw": q, "category": q.get("question_type", "unknown")}
            for q in ds
        ]
        sampled = stratified_sample(temp_questions, sample, seed=seed)
        raw_questions = [q["raw"] for q in sampled]
        console.print(f"  Sampled questions: {len(raw_questions)}")

    # Collect session IDs needed for sampled questions
    needed_sessions: set[str] = set()
    for q in raw_questions:
        needed_sessions.update(q.get("haystack_session_ids", []))
    console.print(f"  Needed sessions: {len(needed_sessions)}")

    # Parse ONLY the sampled questions (filters conversations internally)
    conversations, questions = _parse_dataset_filtered(raw_questions, needed_sessions)

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
