# TribalMemory Benchmark Suite

Independent benchmark harness for evaluating conversational memory systems.

## Supported Benchmarks

| Benchmark | Questions | Focus |
|-----------|-----------|-------|
| [LongMemEval](https://arxiv.org/abs/2410.10813) | 500 | Multi-session long-term memory |
| [ConvoMem](https://arxiv.org/abs/2511.10523) | 75,336 | User facts, preferences, temporal, abstention |

## Supported Providers

- **TribalMemory** (primary)
- Mem0 (planned)
- Zep (planned)
- Supermemory (planned)

## Quick Start

```bash
# Install
pip install -e .

# Download datasets
python -m benchmarks.longmemeval.download
python -m benchmarks.convomem.download

# Run benchmarks
python -m benchmarks.longmemeval --sample 50    # CI: fast
python -m benchmarks.longmemeval --sample 150   # Nightly
python -m benchmarks.longmemeval                 # Full (500)

python -m benchmarks.convomem --sample 500      # CI
python -m benchmarks.convomem --sample 2500     # Nightly
python -m benchmarks.convomem                    # Full (75k)
```

## Configuration

```bash
export TRIBALMEMORY_URL=http://127.0.0.1:18790
# or for other providers:
export MEM0_API_KEY=...
```

## Results

Results are written to `results/` and can be published to `docs/benchmark-results.md` in the main TribalMemory repo.

## Sample Strategies

- **CI (fast)**: Stratified sample across categories, ~5 min
- **Nightly**: Larger sample, ~15-30 min
- **Release**: Full dataset, thorough validation

## License

MIT
