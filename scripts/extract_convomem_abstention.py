#!/usr/bin/env python3
"""Extract abstention cases from ConvoMem dataset.

These are cases where the correct answer is "I don't know" or similar,
testing that the system correctly does NOT hallucinate.
"""

import json
import yaml
from pathlib import Path
from datasets import load_dataset
from collections import defaultdict


def extract_abstention_cases(limit: int = 100) -> list[dict]:
    """Extract abstention category cases from ConvoMem."""
    
    print("Loading ConvoMem dataset...")
    ds = load_dataset("Salesforce/ConvoMem", split="test")
    
    abstention_cases = []
    category_counts = defaultdict(int)
    
    for example in ds:
        category = example.get("category", "unknown")
        category_counts[category] += 1
        
        # Look for abstention category
        if category.lower() in ["abstention", "abstain", "no_answer", "unanswerable"]:
            abstention_cases.append(example)
            if len(abstention_cases) >= limit:
                break
    
    print(f"\nCategory distribution:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")
    
    print(f"\nExtracted {len(abstention_cases)} abstention cases")
    return abstention_cases


def convert_to_scenario(case: dict, idx: int) -> dict:
    """Convert a ConvoMem case to our scenario format."""
    
    # Extract conversation
    conv = case.get("conversation", case.get("messages", []))
    messages = []
    for msg in conv:
        if isinstance(msg, dict):
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })
    
    scenario = {
        "name": f"abstention_{idx:03d}",
        "description": f"No relevant information exists for this query",
        "category": "negative/abstention",
        "source": "ConvoMem",
        
        "conversations": [{
            "session": 1,
            "messages": messages,
        }],
        
        "task": {
            "query": case.get("question", ""),
            "expected_behavior": {
                "should_retrieve": False,
                "correct_response": "acknowledge_uncertainty",
            },
            "success": {
                "indicates_uncertainty": True,
                "does_not_hallucinate": True,
            },
        },
        
        "ground_truth": {
            "answer": case.get("answer", ""),
            "category": case.get("category", ""),
        },
    }
    
    return scenario


def main():
    output_dir = Path(__file__).parent.parent / "scenarios" / "negative" / "abstention"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract cases
    cases = extract_abstention_cases(limit=50)
    
    if not cases:
        print("\nNo abstention cases found. Checking available categories...")
        ds = load_dataset("Salesforce/ConvoMem", split="test")
        categories = set()
        for ex in ds:
            categories.add(ex.get("category", "unknown"))
        print(f"Available categories: {categories}")
        return
    
    # Convert and save
    scenarios = []
    for idx, case in enumerate(cases):
        scenario = convert_to_scenario(case, idx)
        scenarios.append(scenario)
        
        # Save individual scenario
        scenario_path = output_dir / f"abstention_{idx:03d}.yaml"
        with open(scenario_path, "w") as f:
            yaml.dump(scenario, f, default_flow_style=False, sort_keys=False)
    
    # Save combined file
    combined_path = output_dir / "_all_abstention.yaml"
    with open(combined_path, "w") as f:
        yaml.dump({"scenarios": scenarios}, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nSaved {len(scenarios)} scenarios to {output_dir}")
    print(f"Combined file: {combined_path}")


if __name__ == "__main__":
    main()
