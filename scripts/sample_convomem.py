#!/usr/bin/env python3
"""Sample and analyze ConvoMem dataset structure without full download."""

import json
from datasets import load_dataset
from collections import defaultdict

def analyze_categories():
    """Stream through dataset to find categories without full load."""
    
    print("Streaming ConvoMem dataset...")
    ds = load_dataset("Salesforce/ConvoMem", split="train", streaming=True)
    
    category_counts = defaultdict(int)
    category_samples = defaultdict(list)
    
    for i, example in enumerate(ds):
        if i >= 5000:  # Sample first 5k
            break
            
        category = example.get("category", "unknown")
        category_counts[category] += 1
        
        # Keep up to 3 samples per category
        if len(category_samples[category]) < 3:
            category_samples[category].append({
                "question": example.get("question", "")[:200],
                "answer": example.get("answer", "")[:200],
            })
        
        if i % 500 == 0:
            print(f"  Processed {i} examples...")
    
    print(f"\n=== Category Distribution (first 5k) ===")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
    
    print(f"\n=== Sample Questions by Category ===")
    for cat, samples in sorted(category_samples.items()):
        print(f"\n[{cat}]")
        for s in samples[:2]:
            print(f"  Q: {s['question'][:100]}...")
            print(f"  A: {s['answer'][:100]}...")
            print()
    
    return category_counts, category_samples


if __name__ == "__main__":
    analyze_categories()
