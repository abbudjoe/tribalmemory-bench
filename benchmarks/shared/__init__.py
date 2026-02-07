"""Shared benchmark infrastructure."""

from .providers import Provider, TribalMemoryProvider, Memory
from .metrics import (
    BenchmarkResult,
    CategoryResult,
    QuestionResult,
    compute_category_results,
    compute_reciprocal_rank,
    compare_results,
)
from .runner import run_benchmark, stratified_sample, chunk_conversations
from .checkers import (
    substring_checker,
    phrase_checker,
    fuzzy_checker,
    abstention_checker,
    create_checker,
    normalize_text,
)
from .scenario_runner import (
    run_scenario,
    run_scenario_suite,
    load_scenario,
    load_scenarios_from_dir,
    ScenarioResult,
    ScenarioSuiteResult,
)

__all__ = [
    # Providers
    "Provider",
    "TribalMemoryProvider",
    "Memory",
    # Metrics
    "BenchmarkResult",
    "CategoryResult",
    "QuestionResult",
    "compute_category_results",
    "compute_reciprocal_rank",
    "compare_results",
    # Runner
    "run_benchmark",
    "stratified_sample",
    "chunk_conversations",
    # Checkers
    "substring_checker",
    "phrase_checker",
    "fuzzy_checker",
    "abstention_checker",
    "create_checker",
    "normalize_text",
    # Scenarios
    "run_scenario",
    "run_scenario_suite",
    "load_scenario",
    "load_scenarios_from_dir",
    "ScenarioResult",
    "ScenarioSuiteResult",
]
