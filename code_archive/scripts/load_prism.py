"""
Load and process the Prism Alignment dataset.
This dataset contains annotator metadata which can be used for filtering.
"""

import os
import yaml
from datasets import load_dataset
from pathlib import Path


def load_prism_dataset(cache_dir=None, filter_by_annotator=None):
    """
    Load the Prism Alignment dataset from Hugging Face.
    
    Args:
        cache_dir: Directory to cache the dataset. If None, uses config.yaml
        filter_by_annotator: Optional annotator ID or list of IDs to filter by
        
    Returns:
        Dataset or DatasetDict containing the Prism dataset
    """
    # Load config if available
    config_path = Path(__file__).parent.parent / "config.yaml"
    if cache_dir is None and config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            cache_dir = config.get('datasets', {}).get('prism', {}).get('cache_dir', './data/prism')
    
    if cache_dir is None:
        cache_dir = './data/prism'
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    print("Loading Prism Alignment dataset...")
    dataset = load_dataset(
        "HannahRoseKirk/prism-alignment",
        cache_dir=cache_dir
    )
    
    print(f"Dataset loaded: {dataset}")
    
    # Display dataset structure
    if hasattr(dataset, 'keys'):
        for split in dataset.keys():
            print(f"\n{split} split:")
            print(f"  Number of examples: {len(dataset[split])}")
            if len(dataset[split]) > 0:
                print(f"  Features: {list(dataset[split][0].keys())}")
                # Show first example
                print(f"  First example keys: {list(dataset[split][0].keys())}")
    else:
        print(f"\nNumber of examples: {len(dataset)}")
        if len(dataset) > 0:
            print(f"Features: {list(dataset[0].keys())}")
    
    # Filter by annotator if requested
    if filter_by_annotator is not None:
        print(f"\nFiltering by annotator: {filter_by_annotator}")
        # This will depend on the actual structure of the dataset
        # Adjust based on actual feature names
        if isinstance(dataset, dict):
            for split in dataset.keys():
                if 'annotator' in dataset[split].features or 'annotator_id' in dataset[split].features:
                    if isinstance(filter_by_annotator, list):
                        dataset[split] = dataset[split].filter(
                            lambda x: x.get('annotator') in filter_by_annotator or 
                                     x.get('annotator_id') in filter_by_annotator
                        )
                    else:
                        dataset[split] = dataset[split].filter(
                            lambda x: x.get('annotator') == filter_by_annotator or 
                                     x.get('annotator_id') == filter_by_annotator
                        )
        else:
            if 'annotator' in dataset.features or 'annotator_id' in dataset.features:
                if isinstance(filter_by_annotator, list):
                    dataset = dataset.filter(
                        lambda x: x.get('annotator') in filter_by_annotator or 
                                 x.get('annotator_id') in filter_by_annotator
                    )
                else:
                    dataset = dataset.filter(
                        lambda x: x.get('annotator') == filter_by_annotator or 
                                 x.get('annotator_id') == filter_by_annotator
                    )
    
    return dataset


def explore_prism_annotators(dataset):
    """
    Explore annotator metadata in the Prism dataset.
    
    Args:
        dataset: The loaded Prism dataset
        
    Returns:
        Dictionary with annotator statistics
    """
    print("\n=== Exploring Annotator Metadata ===")
    
    annotator_info = {}
    
    # Handle both DatasetDict and Dataset
    if hasattr(dataset, 'keys'):
        splits = dataset.keys()
    else:
        splits = ['default']
        dataset = {'default': dataset}
    
    for split in splits:
        print(f"\n{split} split:")
        split_data = dataset[split]
        
        # Try to find annotator-related columns
        features = split_data.features.keys()
        annotator_cols = [col for col in features if 'annotator' in col.lower() or 'metadata' in col.lower()]
        
        if annotator_cols:
            print(f"  Found annotator-related columns: {annotator_cols}")
            for col in annotator_cols:
                unique_values = set(split_data[col])
                print(f"  {col}: {len(unique_values)} unique values")
                annotator_info[col] = {
                    'unique_count': len(unique_values),
                    'values': list(unique_values)[:10]  # Show first 10
                }
        else:
            print(f"  No obvious annotator columns found. Available columns: {list(features)}")
    
    return annotator_info


if __name__ == "__main__":
    # Load the dataset
    dataset = load_prism_dataset()
    
    # Explore annotator metadata
    annotator_info = explore_prism_annotators(dataset)
    
    # Save summary
    import json
    output_path = Path(__file__).parent.parent / "results" / "prism_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'dataset_info': str(dataset),
            'annotator_info': annotator_info
        }, f, indent=2, default=str)
    
    print(f"\nSummary saved to {output_path}")

