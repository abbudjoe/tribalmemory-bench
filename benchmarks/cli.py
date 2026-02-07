"""TribalMemory Benchmark CLI."""

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="TribalMemory Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tribench longmemeval --sample 50          # Quick CI run
  tribench convomem --sample 500            # CI sample
  tribench longmemeval                      # Full benchmark
  tribench convomem --output results/       # Save results
        """,
    )
    
    parser.add_argument(
        "benchmark",
        choices=["longmemeval", "convomem", "all"],
        help="Benchmark to run",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Sample size (stratified by category)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--provider-url",
        type=str,
        help="TribalMemory server URL (default: http://127.0.0.1:18790)",
    )
    parser.add_argument(
        "--instance",
        type=str,
        help="Instance ID for isolation",
    )
    
    args = parser.parse_args()
    
    if args.benchmark == "longmemeval":
        from .longmemeval.harness import run_longmemeval
        from .shared.providers import TribalMemoryProvider
        
        provider = TribalMemoryProvider(
            base_url=args.provider_url,
            instance=args.instance or "longmemeval",
        )
        asyncio.run(run_longmemeval(
            provider=provider,
            sample=args.sample,
            output_dir=Path(args.output),
        ))
        
    elif args.benchmark == "convomem":
        from .convomem.harness import run_convomem
        from .shared.providers import TribalMemoryProvider
        
        provider = TribalMemoryProvider(
            base_url=args.provider_url,
            instance=args.instance or "convomem",
        )
        asyncio.run(run_convomem(
            provider=provider,
            sample=args.sample,
            output_dir=Path(args.output),
        ))
        
    elif args.benchmark == "all":
        from .longmemeval.harness import run_longmemeval
        from .convomem.harness import run_convomem
        from .shared.providers import TribalMemoryProvider
        
        console.print("[bold]Running all benchmarks...[/bold]\n")
        
        # LongMemEval
        provider = TribalMemoryProvider(
            base_url=args.provider_url,
            instance="longmemeval",
        )
        asyncio.run(run_longmemeval(
            provider=provider,
            sample=args.sample,
            output_dir=Path(args.output),
        ))
        
        # ConvoMem
        provider = TribalMemoryProvider(
            base_url=args.provider_url,
            instance="convomem",
        )
        asyncio.run(run_convomem(
            provider=provider,
            sample=args.sample,
            output_dir=Path(args.output),
        ))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
