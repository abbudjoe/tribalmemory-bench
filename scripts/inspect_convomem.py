#!/usr/bin/env python3
"""Inspect ConvoMem dataset structure."""

from datasets import load_dataset

print("Loading ConvoMem info...")
ds = load_dataset("Salesforce/ConvoMem", streaming=True)

print(f"\nAvailable splits: {list(ds.keys())}")

# Get first example from train
train_iter = iter(ds['train'])
example = next(train_iter)

print(f"\nExample keys: {list(example.keys())}")
print(f"\nExample structure:")
for key, value in example.items():
    if isinstance(value, (str, int, float, bool)):
        print(f"  {key}: {type(value).__name__} = {str(value)[:100]}")
    elif isinstance(value, list):
        print(f"  {key}: list[{len(value)} items]")
        if value and isinstance(value[0], dict):
            print(f"    item keys: {list(value[0].keys())}")
    elif isinstance(value, dict):
        print(f"  {key}: dict with keys {list(value.keys())}")
    else:
        print(f"  {key}: {type(value).__name__}")
