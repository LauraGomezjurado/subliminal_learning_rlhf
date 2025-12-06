"""
Explore PRISM and OpinionsQA datasets to understand their structure.
"""

import os
from datasets import load_dataset
from pathlib import Path
import json
from collections import Counter

def explore_prism():
    """Explore the PRISM dataset structure."""
    print("=" * 80)
    print("EXPLORING PRISM DATASET")
    print("=" * 80)
    
    # Available configs
    configs = ['survey', 'conversations', 'utterances', 'metadata']
    
    for config_name in configs:
        print(f"\n{'='*80}")
        print(f"CONFIG: {config_name}")
        print(f"{'='*80}")
        
        try:
            dataset = load_dataset(
                "HannahRoseKirk/prism-alignment",
                config_name,
                cache_dir="./data/prism"
            )
            
            print(f"\nDataset structure: {dataset}")
            
            # Explore each split
            for split_name in dataset.keys():
                print(f"\n--- Split: {split_name} ---")
                split = dataset[split_name]
                print(f"Number of examples: {len(split)}")
                print(f"Features: {list(split.features.keys())}")
                
                # Show first example
                if len(split) > 0:
                    print(f"\nFirst example:")
                    first_example = split[0]
                    for key, value in first_example.items():
                        if isinstance(value, (str, int, float, bool)):
                            print(f"  {key}: {value}")
                        elif isinstance(value, list) and len(value) > 0:
                            print(f"  {key}: [list with {len(value)} items]")
                            if isinstance(value[0], (str, int, float)):
                                print(f"    First item: {value[0]}")
                        else:
                            print(f"  {key}: {type(value).__name__}")
                    
                    # Check for annotator/demographic info
                    demographic_fields = [k for k in first_example.keys() 
                                         if any(term in k.lower() for term in 
                                               ['annotator', 'demographic', 'region', 
                                                'age', 'gender', 'country', 'political'])]
                    
                    if demographic_fields:
                        print(f"\nDemographic/annotator fields found: {demographic_fields}")
                        for field in demographic_fields:
                            unique_vals = set([ex[field] for ex in split[:100] if ex.get(field) is not None])
                            print(f"  {field}: {len(unique_vals)} unique values (sample from first 100)")
                            print(f"    Sample values: {list(unique_vals)[:5]}")
                
        except Exception as e:
            print(f"Error loading config {config_name}: {e}")


def explore_opinions_qa():
    """Explore the OpinionsQA dataset structure."""
    print("\n" + "=" * 80)
    print("EXPLORING OPINIONSQA DATASET")
    print("=" * 80)
    
    try:
        dataset = load_dataset(
            "Anthropic/llm_global_opinions",
            cache_dir="./data/opinionsqa"
        )
        
        print(f"\nDataset structure: {dataset}")
        
        for split_name in dataset.keys():
            print(f"\n--- Split: {split_name} ---")
            split = dataset[split_name]
            print(f"Number of examples: {len(split)}")
            print(f"Features: {list(split.features.keys())}")
            
            if len(split) > 0:
                print(f"\nFirst 3 examples:")
                for i in range(min(3, len(split))):
                    example = split[i]
                    print(f"\nExample {i+1}:")
                    for key, value in example.items():
                        if isinstance(value, str):
                            display_val = value[:200] + "..." if len(value) > 200 else value
                            print(f"  {key}: {display_val}")
                        else:
                            print(f"  {key}: {value}")
        
        # Try alternative dataset names
        print("\nTrying alternative OpinionsQA sources...")
        
    except Exception as e:
        print(f"Error loading OpinionsQA: {e}")
        print("\nTrying alternative dataset name: 'tatsu-lab/opinions_qa'")
        
        try:
            dataset = load_dataset(
                "tatsu-lab/opinions_qa", 
                cache_dir="./data/opinionsqa"
            )
            print(f"\nDataset structure: {dataset}")
            
            for split_name in dataset.keys():
                print(f"\n--- Split: {split_name} ---")
                split = dataset[split_name]
                print(f"Number of examples: {len(split)}")
                print(f"Features: {list(split.features.keys())}")
                
                if len(split) > 0:
                    print(f"\nFirst example:")
                    example = split[0]
                    for key, value in example.items():
                        if isinstance(value, str):
                            display_val = value[:200] + "..." if len(value) > 200 else value
                            print(f"  {key}: {display_val}")
                        else:
                            print(f"  {key}: {value}")
        except Exception as e2:
            print(f"Also failed: {e2}")


if __name__ == "__main__":
    # Create data directories
    os.makedirs("./data/prism", exist_ok=True)
    os.makedirs("./data/opinionsqa", exist_ok=True)
    
    # Explore datasets
    explore_prism()
    explore_opinions_qa()
    
    print("\n" + "=" * 80)
    print("EXPLORATION COMPLETE")
    print("=" * 80)

