"""Scenario-based evaluation runner."""

import asyncio
import time
import yaml
import aiofiles
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from rich.console import Console

from .providers import Provider
from .checkers import normalize_text

console = Console()


@dataclass
class ScenarioResult:
    """Result for a single scenario."""
    name: str
    category: str
    passed: bool
    failure_mode: Optional[str] = None
    failure_description: Optional[str] = None
    latency_ms: float = 0.0
    retrieved: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


@dataclass
class ScenarioSuiteResult:
    """Results for a suite of scenarios."""
    suite_name: str
    total: int
    passed: int
    pass_rate: float
    results: list[ScenarioResult]
    by_category: dict[str, dict] = field(default_factory=dict)
    avg_latency_ms: float = 0.0


async def load_scenario(path: Path) -> dict:
    """Load a scenario from YAML file (async)."""
    async with aiofiles.open(path) as f:
        content = await f.read()
        return yaml.safe_load(content)


async def load_scenarios_from_dir(dir_path: Path) -> list[dict]:
    """Load all scenarios from a directory (async)."""
    scenarios = []
    for path in sorted(dir_path.glob("**/*.yaml")):
        # Skip combined files
        if path.name.startswith("_"):
            continue
        try:
            scenario = await load_scenario(path)
            if scenario:
                scenario["_path"] = str(path)
                scenarios.append(scenario)
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to load {path}: {e}[/yellow]")
    return scenarios


async def run_scenario(
    scenario: dict,
    provider: Provider,
) -> ScenarioResult:
    """
    Run a single scenario.
    
    1. Ingest conversations
    2. Run task query
    3. Check expected behavior
    4. Classify failure if failed
    """
    name = scenario.get("name", "unnamed")
    category = scenario.get("category", "unknown")
    
    # Phase 1: Ingest conversations
    conversations = scenario.get("conversations", [])
    for conv in conversations:
        messages = conv.get("messages", [])
        session_id = conv.get("session", "")
        
        # Chunk messages into pairs
        for i in range(0, len(messages), 2):
            chunk = messages[i:i+2]
            content = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}"
                for m in chunk
            )
            await provider.store(content, context=f"session:{session_id}")
    
    # Phase 2: Run task query
    task = scenario.get("task", {})
    query = task.get("query", "")
    
    start = time.time()
    memories = await provider.recall(query, limit=10)
    latency_ms = (time.time() - start) * 1000
    
    retrieved = [m.content for m in memories]
    
    # Phase 3: Check expected behavior
    expected = task.get("expected_behavior", {})
    success_criteria = task.get("success", {})
    failure_modes = scenario.get("failure_modes", [])
    
    passed, failure_mode, failure_desc = check_expected_behavior(
        expected=expected,
        success=success_criteria,
        failure_modes=failure_modes,
        retrieved=retrieved,
    )
    
    return ScenarioResult(
        name=name,
        category=category,
        passed=passed,
        failure_mode=failure_mode,
        failure_description=failure_desc,
        latency_ms=latency_ms,
        retrieved=retrieved[:3],  # Keep top 3 for debugging
        details={
            "query": query,
            "expected_behavior": expected,
        }
    )


def check_expected_behavior(
    expected: dict,
    success: dict,
    failure_modes: list[dict],
    retrieved: list[str],
) -> tuple[bool, Optional[str], Optional[str]]:
    """
    Check if retrieved memories match expected behavior.
    
    Returns:
        (passed, failure_mode, failure_description)
    """
    combined = " ".join(normalize_text(r) for r in retrieved)
    combined_raw = " ".join(retrieved)
    
    # Check should_retrieve
    should_retrieve = expected.get("should_retrieve", True)
    
    if not should_retrieve:
        # Negative test: should NOT find relevant content
        if not retrieved:
            return (True, None, None)
        # Check if retrieved content is actually relevant (simple heuristic).
        # 30 chars ≈ 5-6 words - too short to be meaningful context.
        # This is a rough filter; semantic relevance checking would be better.
        if len(combined) < 30:
            return (True, None, None)
        # Found content when shouldn't have
        return (False, "false_positive", "Retrieved content when should not have")
    
    # Positive test: should find and use relevant content
    if not retrieved:
        return (False, "no_retrieval", "No memories retrieved")
    
    # Check success criteria
    response_indicates = success.get(
        "response_indicates", success.get("contains", [])
    )
    response_not_indicates = success.get(
        "response_does_not_indicate", success.get("not_contains", [])
    )
    
    # Check positive indicators
    found_positive = False
    for indicator in response_indicates:
        if normalize_text(indicator) in combined:
            found_positive = True
            break
    
    if response_indicates and not found_positive:
        # Try to classify failure mode
        for fm in failure_modes:
            fm_type = fm.get("type", "unknown")
            # Simple heuristic: check if failure mode keywords appear
            if fm_type == "stale_retrieval":
                should_ignore = expected.get("should_ignore", [])
                for ignore in should_ignore:
                    if isinstance(ignore, str) and normalize_text(ignore) in combined:
                        return (False, fm_type, fm.get("description", ""))
        
        return (False, "missing_expected", "Expected indicators not found in retrieved")
    
    # Check negative indicators (things that should NOT appear)
    for indicator in response_not_indicates:
        if normalize_text(indicator) in combined:
            # Found something that shouldn't be there
            for fm in failure_modes:
                if fm.get("type") == "stale_retrieval":
                    return (False, "stale_retrieval", fm.get("description", ""))
            return (False, "unexpected_content", f"Found '{indicator}' which should not appear")
    
    return (True, None, None)


async def run_scenario_suite(
    scenarios: list[dict],
    provider: Provider,
    show_progress: bool = True,
) -> ScenarioSuiteResult:
    """
    Run a suite of scenarios.
    """
    from rich.progress import Progress
    from collections import defaultdict
    
    results: list[ScenarioResult] = []
    by_category: dict[str, list[ScenarioResult]] = defaultdict(list)
    
    console.print(f"[bold blue]Running {len(scenarios)} scenarios[/bold blue]")
    
    with Progress() as progress:
        task = progress.add_task("Scenarios", total=len(scenarios))
        
        for scenario in scenarios:
            result = await run_scenario(scenario, provider)
            results.append(result)
            by_category[result.category].append(result)
            progress.advance(task)
    
    # Compute summary
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    avg_latency = sum(r.latency_ms for r in results) / total if total > 0 else 0
    
    # Category breakdown
    category_summary = {}
    for cat, cat_results in by_category.items():
        cat_total = len(cat_results)
        cat_passed = sum(1 for r in cat_results if r.passed)
        category_summary[cat] = {
            "total": cat_total,
            "passed": cat_passed,
            "pass_rate": cat_passed / cat_total if cat_total > 0 else 0,
        }
    
    # Print summary
    console.print(f"\n[bold green]Results:[/bold green]")
    console.print(f"  Overall: {passed}/{total} passed ({passed/total:.1%})")
    console.print(f"  Avg Latency: {avg_latency:.1f}ms")
    console.print("\n  By Category:")
    for cat, summary in sorted(category_summary.items()):
        console.print(
            f"    {cat}: {summary['passed']}/{summary['total']} "
            f"({summary['pass_rate']:.1%})"
        )
    
    # Show failures
    failures = [r for r in results if not r.passed]
    if failures:
        console.print(f"\n  [red]Failures ({len(failures)}):[/red]")
        for f in failures[:5]:  # Show first 5
            console.print(f"    - {f.name}: {f.failure_mode} - {f.failure_description}")
        if len(failures) > 5:
            console.print(f"    ... and {len(failures) - 5} more")
    
    return ScenarioSuiteResult(
        suite_name="scenarios",
        total=total,
        passed=passed,
        pass_rate=passed / total if total > 0 else 0,
        results=results,
        by_category=category_summary,
        avg_latency_ms=avg_latency,
    )
