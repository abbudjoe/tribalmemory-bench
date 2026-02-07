#!/usr/bin/env python3
"""Inspect ConvoMem dataset structure with trust_remote_code."""

from datasets import load_dataset

print("Loading ConvoMem...")
try:
    # Try with trust_remote_code
    ds = load_dataset("Salesforce/ConvoMem", trust_remote_code=True)
    print(f"Splits: {list(ds.keys())}")
    print(f"Train size: {len(ds['train'])}")
    
    example = ds['train'][0]
    print(f"\nExample keys: {list(example.keys())}")
    
except Exception as e:
    print(f"Error with trust_remote_code: {e}")
    
    # Try loading raw parquet files
    print("\nTrying parquet approach...")
    import requests
    
    # Check repo contents
    api_url = "https://huggingface.co/api/datasets/Salesforce/ConvoMem/parquet"
    resp = requests.get(api_url)
    if resp.ok:
        print(f"Parquet files: {resp.json()}")
