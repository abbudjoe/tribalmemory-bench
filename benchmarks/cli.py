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
  tribench scenarios negative/              # Run negative scenarios
  tribench scenarios temporal/ --provider-url http://localhost:18790
        """,
    )
    
    parser.add_argument(
        "benchmark",
        help="Benchmark to run: longmemeval, convomem, scenarios, all",
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Path for scenarios (e.g., negative/, temporal/)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Sample size (stratified by category)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
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
        help="Instance ID for isolation (default: auto-generated UUID)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Batch size for ingestion (default: 20)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Query concurrency (default: 10)",
    )
    
    args = parser.parse_args()
    
    if args.benchmark == "longmemeval":
        asyncio.run(run_longmemeval_cmd(args))
        
    elif args.benchmark == "convomem":
        asyncio.run(run_convomem_cmd(args))
        
    elif args.benchmark == "scenarios":
        if not args.path:
            console.print("[red]Error: scenarios requires a path argument[/red]")
            console.print("Usage: tribench scenarios negative/")
            return 1
        asyncio.run(run_scenarios_cmd(args))
        
    elif args.benchmark == "all":
        asyncio.run(run_all_cmd(args))
        
    else:
        console.print(f"[red]Unknown benchmark: {args.benchmark}[/red]")
        console.print("Available: longmemeval, convomem, scenarios, all")
        return 1
    
    return 0


async def run_longmemeval_cmd(args):
    """Run LongMemEval benchmark."""
    from .longmemeval.harness import run_longmemeval
    from .shared.providers import TribalMemoryProvider
    
    async with TribalMemoryProvider(
        base_url=args.provider_url,
        instance=args.instance or "longmemeval",
    ) as provider:
        await run_longmemeval(
            provider=provider,
            sample=args.sample,
            output_dir=Path(args.output),
        )


async def run_convomem_cmd(args):
    """Run ConvoMem benchmark."""
    from .convomem.harness import run_convomem
    from .shared.providers import TribalMemoryProvider
    
    async with TribalMemoryProvider(
        base_url=args.provider_url,
        instance=args.instance or "convomem",
    ) as provider:
        await run_convomem(
            provider=provider,
            sample=args.sample,
            output_dir=Path(args.output),
        )


async def run_scenarios_cmd(args):
    """Run scenario-based evaluation."""
    from .shared.providers import TribalMemoryProvider
    from .shared.scenario_runner import load_scenarios_from_dir, run_scenario_suite
    
    # Find scenarios directory
    scenarios_base = Path(__file__).parent.parent / "scenarios"
    scenario_path = scenarios_base / args.path
    
    if not scenario_path.exists():
        console.print(f"[red]Scenario path not found: {scenario_path}[/red]")
        return
    
    scenarios = load_scenarios_from_dir(scenario_path)
    
    if not scenarios:
        console.print(f"[yellow]No scenarios found in {scenario_path}[/yellow]")
        return
    
    async with TribalMemoryProvider(
        base_url=args.provider_url,
        instance=args.instance,  # Use UUID for isolation
    ) as provider:
        result = await run_scenario_suite(scenarios, provider)
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    result_path = output_dir / f"scenarios-{args.path.replace('/', '-')}.json"
    with open(result_path, "w") as f:
        json.dump({
            "suite": result.suite_name,
            "total": result.total,
            "passed": result.passed,
            "pass_rate": result.pass_rate,
            "by_category": result.by_category,
            "avg_latency_ms": result.avg_latency_ms,
        }, f, indent=2)
    
    console.print(f"\n[green]Results saved to {result_path}[/green]")


async def run_all_cmd(args):
    """Run all benchmarks."""
    console.print("[bold]Running all benchmarks...[/bold]\n")
    
    await run_longmemeval_cmd(args)
    await run_convomem_cmd(args)


if __name__ == "__main__":
    sys.exit(main())
