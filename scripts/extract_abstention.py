#!/usr/bin/env python3
"""Extract abstention cases from cached ConvoMem dataset."""

import json
import yaml
from pathlib import Path
from glob import glob

CACHE_DIR = Path.home() / ".cache/huggingface/hub/datasets--Salesforce--ConvoMem"
OUTPUT_DIR = Path(__file__).parent.parent / "scenarios" / "negative" / "abstention"


def find_abstention_files():
    """Find all abstention evidence JSON files."""
    pattern = str(CACHE_DIR / "snapshots/*/core_benchmark/pre_mixed_testcases/abstention_evidence/**/*.json")
    return glob(pattern, recursive=True)


def parse_convomem_json(filepath: str) -> list[dict]:
    """Parse a ConvoMem JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "evidenceItems" in data:
        return data["evidenceItems"]
    return []


def convert_to_scenario(item: dict, idx: int) -> dict:
    """Convert ConvoMem item to our scenario format."""
    
    # Extract messages from conversations
    messages = []
    for conv in item.get("conversations", []):
        for msg in conv.get("messages", []):
            messages.append({
                "role": "user" if msg.get("speaker", "").lower() == "user" else "assistant",
                "content": msg.get("text", ""),
            })
    
    return {
        "name": f"convomem_abstention_{idx:03d}",
        "description": "Question cannot be answered from conversation history",
        "category": "negative/abstention",
        "source": "ConvoMem",
        
        "conversations": [{
            "session": 1,
            "messages": messages[:20],  # Limit for readability
        }],
        
        "task": {
            "query": item.get("question", ""),
            "expected_behavior": {
                "should_retrieve": False,
                "reasoning": "Information not present in conversation history",
            },
            "success": {
                "acknowledges_uncertainty": True,
                "does_not_hallucinate": True,
            },
        },
        
        "ground_truth": {
            "answer": item.get("answer", ""),
            "evidence_messages": [
                {"speaker": e.get("speaker"), "text": e.get("text", "")[:100]}
                for e in item.get("message_evidences", [])[:3]
            ],
        },
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    files = find_abstention_files()
    print(f"Found {len(files)} abstention files")
    
    all_items = []
    for filepath in files:
        items = parse_convomem_json(filepath)
        all_items.extend(items)
    
    print(f"Total abstention items: {len(all_items)}")
    
    # Convert first 50 to scenarios
    scenarios = []
    for idx, item in enumerate(all_items[:50]):
        scenario = convert_to_scenario(item, idx)
        scenarios.append(scenario)
        
        # Save individual file
        filepath = OUTPUT_DIR / f"convomem_{idx:03d}.yaml"
        with open(filepath, "w") as f:
            yaml.dump(scenario, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    # Save combined
    combined_path = OUTPUT_DIR / "_all.yaml"
    with open(combined_path, "w") as f:
        yaml.dump({"scenarios": scenarios}, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nSaved {len(scenarios)} scenarios to {OUTPUT_DIR}")
    print(f"\nSample question: {scenarios[0]['task']['query'][:100]}...")
    print(f"Expected behavior: {scenarios[0]['task']['expected_behavior']}")


if __name__ == "__main__":
    main()
