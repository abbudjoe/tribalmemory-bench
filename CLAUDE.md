# CLAUDE.md - tribalmemory-bench

## Project Overview
Benchmark suite for memory providers (TribalMemory, Mem0, Zep, etc.) using standardized evaluation benchmarks (LongMemEval, LoCoMo).

## Tech Stack
- TypeScript + Bun
- GPT-4o for answer generation and evaluation

## PR Review Loop (MANDATORY)

**All changes must follow this flow. No exceptions.**

1. Create feature branch from main
2. Make changes, commit
3. Push branch, open PR
4. **Post PR comment: `@claude review this PR`** ← REQUIRED, DO NOT SKIP
5. **Post PR comment: `@codex review this PR`** ← REQUIRED, DO NOT SKIP
6. **Wait 5 minutes, then check PR for review comments from both reviewers**
7. **Address ALL comments from both reviews** (every single item)
8. Commit fixes, **push**
9. **Post PR comment: `@claude review this PR`** ← REQUIRED AFTER EVERY PUSH
10. **Post PR comment: `@codex review this PR`** ← REQUIRED AFTER EVERY PUSH
11. **Wait 5 minutes, check for new review comments**
12. Issues remaining? → Go to step 7 (address all comments again)
13. When clean: Comment `@abbudjoe ready for merge`
14. Joe reviews and merges

**CRITICAL:**
- ❌ No direct commits to main
- ❌ No merges without review
- ❌ No skipping the `@claude review this PR` comment (required after EVERY push)
- ❌ No skipping the `@codex review this PR` comment (required after EVERY push)
- ❌ Do NOT put review trigger comments in commit messages
- ❌ Do NOT skip review items marked "low priority", "nice to have", or "suggestion"
- ✅ Comment after **every push** to trigger re-review
- ✅ **Address EVERY comment** — partial fixes are not acceptable
