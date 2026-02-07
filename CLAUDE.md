# CLAUDE.md - TribalMemory Benchmark Suite

## Project Overview
Independent benchmark harness for evaluating conversational memory systems. Supports LongMemEval, ConvoMem, and scenario-based evaluation.

## Tech Stack
- Python 3.10+
- pytest + pytest-asyncio for testing
- httpx for async HTTP
- rich for CLI output
- datasets (HuggingFace) for benchmark loading

## Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run benchmarks
tribench longmemeval --sample 50     # Quick CI run
tribench convomem --sample 500       # CI sample
tribench longmemeval                  # Full benchmark

# Run scenarios
tribench scenarios negative/         # Negative test scenarios
tribench scenarios temporal/         # Temporal reasoning scenarios
```

## TDD Requirements (MANDATORY)

1. **RED** — Write failing test first
2. **GREEN** — Minimal code to pass
3. **REFACTOR** — Clean up while green

Every feature needs tests. Every bug fix needs a failing regression test first.

## Code Style

- Line length: 100 chars
- Use type hints
- Async functions for all I/O
- Dataclasses for data structures
- Abstract base classes for interfaces

## PR Review Loop (MANDATORY)

**All changes must follow this flow. No exceptions.**

1. Create feature branch from main
2. Make changes, commit
3. Push branch, open PR
4. **Post PR comment: `@claude review this PR`** ← REQUIRED, DO NOT SKIP  
   ⚠️ **Note:** This is a comment on the PR (not the commit message)
5. **Wait 5 minutes, then check PR for review comments**
6. **Address ALL comments from the review** (every single item)
7. Commit fixes, **push**
8. **Post PR comment: `@claude review this PR`** ← REQUIRED AFTER EVERY PUSH
9. **Wait 5 minutes, check for new review comments**
10. Issues remaining? → Go to step 6 (address all comments again)
11. When clean: Comment `@abbudjoe ready for merge`
12. Joe reviews and merges

**CRITICAL:**
- ❌ No direct commits to main (except hotfixes approved by Joe)
- ❌ No merges without Claude Code review
- ❌ No skipping the `@claude review this PR` comment (required after EVERY push)
- ❌ Do NOT put `@claude review this PR` in commit messages
- ❌ Do NOT rely on automatic GitHub Action triggers
- ✅ The comment must be **standalone** on the PR (not combined with other text)
- ✅ Comment after **every push** to trigger re-review
- ✅ **Check back after 5 minutes** to view and address all review comments
- ✅ **Address EVERY comment** — partial fixes are not acceptable

## Project Structure

```
tribalmemory-bench/
├── benchmarks/
│   ├── shared/           # Shared provider/runner/metrics
│   │   ├── providers.py  # Memory provider abstractions
│   │   ├── runner.py     # Benchmark orchestration
│   │   └── metrics.py    # Result metrics
│   ├── longmemeval/      # LongMemEval harness
│   └── convomem/         # ConvoMem harness
├── scenarios/            # Task-completion scenarios
│   ├── positive/         # Should retrieve + use
│   ├── negative/         # Should NOT retrieve/use
│   ├── temporal/         # Change tracking
│   ├── privacy/          # Isolation tests
│   └── degradation/      # Scale tests
├── scripts/              # Dataset extraction tools
├── tests/                # Test files
└── results/              # Benchmark output
```

## Adding a New Provider

1. Implement `Provider` abstract class in `benchmarks/shared/providers.py`
2. Add tests in `tests/test_providers.py`
3. Register in CLI (`benchmarks/cli.py`)

## Adding a New Benchmark

1. Create directory under `benchmarks/`
2. Implement `download_dataset()`, `parse_dataset()`, `answer_checker()`
3. Add harness using `run_benchmark()` from shared runner
4. Add tests

## Checklist Before PR

1. ✅ All tests pass (`pytest`)
2. ✅ No type errors
3. ✅ Code formatted
4. ✅ New tests for new features
5. ✅ Documentation updated if needed
6. ✅ `@claude review this PR` comment posted
