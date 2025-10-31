"""
Load and process the Opinions QA dataset.
Dataset repository: https://github.com/tatsu-lab/opinions_qa
"""

import os
import yaml
import subprocess
from pathlib import Path
import json


def clone_opinions_qa_repo(cache_dir=None):
    """
    Clone the Opinions QA repository if not already present.
    
    Args:
        cache_dir: Directory to store the cloned repository. If None, uses config.yaml
        
    Returns:
        Path to the cloned repository
    """
    config_path = Path(__file__).parent.parent / "config.yaml"
    if cache_dir is None and config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            cache_dir = config.get('datasets', {}).get('opinions_qa', {}).get('cache_dir', './data/opinions_qa')
    
    if cache_dir is None:
        cache_dir = './data/opinions_qa'
    
    repo_dir = Path(cache_dir) / "opinions_qa"
    repo_url = "https://github.com/tatsu-lab/opinions_qa.git"
    
    if repo_dir.exists() and (repo_dir / ".git").exists():
        print(f"Repository already exists at {repo_dir}")
        return repo_dir
    
    print(f"Cloning Opinions QA repository to {repo_dir}...")
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        subprocess.run(
            ["git", "clone", repo_url, str(repo_dir)],
            check=True,
            capture_output=True
        )
        print(f"Repository cloned successfully to {repo_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
        print(f"stdout: {e.stdout.decode()}")
        print(f"stderr: {e.stderr.decode()}")
        raise
    
    return repo_dir


def load_opinions_qa_dataset(cache_dir=None):
    """
    Load the Opinions QA dataset.
    
    Args:
        cache_dir: Directory where the dataset is stored. If None, uses config.yaml
        
    Returns:
        Dictionary containing dataset paths and information
    """
    repo_dir = clone_opinions_qa_repo(cache_dir)
    
    print(f"\nExploring Opinions QA repository structure...")
    
    dataset_info = {
        'repo_path': str(repo_dir),
        'files': [],
        'data_files': []
    }
    
    # List all files in the repository
    for root, dirs, files in os.walk(repo_dir):
        # Skip .git directory
        dirs[:] = [d for d in dirs if d != '.git']
        
        for file in files:
            file_path = Path(root) / file
            rel_path = file_path.relative_to(repo_dir)
            
            if file.endswith(('.json', '.jsonl', '.csv', '.tsv', '.txt')):
                dataset_info['data_files'].append(str(rel_path))
            else:
                dataset_info['files'].append(str(rel_path))
    
    print(f"\nFound {len(dataset_info['data_files'])} potential data files:")
    for data_file in dataset_info['data_files'][:20]:  # Show first 20
        print(f"  - {data_file}")
    if len(dataset_info['data_files']) > 20:
        print(f"  ... and {len(dataset_info['data_files']) - 20} more")
    
    # Try to find and load common data files
    data = {}
    
    # Look for common dataset file patterns
    common_patterns = ['data', 'dataset', 'train', 'test', 'val', 'dev']
    for pattern in common_patterns:
        for data_file in dataset_info['data_files']:
            if pattern in data_file.lower():
                file_path = repo_dir / data_file
                try:
                    if data_file.endswith('.json'):
                        with open(file_path, 'r') as f:
                            content = json.load(f)
                            data[data_file] = content
                            print(f"\nLoaded {data_file}")
                    elif data_file.endswith('.jsonl'):
                        items = []
                        with open(file_path, 'r') as f:
                            for line in f:
                                items.append(json.loads(line))
                        data[data_file] = items
                        print(f"\nLoaded {data_file} ({len(items)} items)")
                except Exception as e:
                    print(f"Could not load {data_file}: {e}")
    
    dataset_info['loaded_data'] = data
    
    return dataset_info


def explore_opinions_qa_structure(dataset_info):
    """
    Explore the structure of the Opinions QA dataset.
    
    Args:
        dataset_info: Dictionary returned by load_opinions_qa_dataset
        
    Returns:
        Dictionary with structure information
    """
    print("\n=== Exploring Opinions QA Structure ===")
    
    structure = {
        'repo_path': dataset_info['repo_path'],
        'data_files_count': len(dataset_info['data_files']),
        'loaded_datasets': {}
    }
    
    for data_file, content in dataset_info.get('loaded_data', {}).items():
        print(f"\n{data_file}:")
        
        if isinstance(content, list):
            print(f"  Type: List with {len(content)} items")
            if len(content) > 0:
                print(f"  First item keys: {list(content[0].keys()) if isinstance(content[0], dict) else 'Not a dict'}")
                structure['loaded_datasets'][data_file] = {
                    'type': 'list',
                    'count': len(content),
                    'sample_keys': list(content[0].keys()) if isinstance(content[0], dict) else None
                }
        elif isinstance(content, dict):
            print(f"  Type: Dictionary")
            print(f"  Top-level keys: {list(content.keys())[:10]}")
            structure['loaded_datasets'][data_file] = {
                'type': 'dict',
                'top_level_keys': list(content.keys())
            }
        else:
            print(f"  Type: {type(content)}")
            structure['loaded_datasets'][data_file] = {
                'type': str(type(content))
            }
    
    return structure


if __name__ == "__main__":
    # Load the dataset
    dataset_info = load_opinions_qa_dataset()
    
    # Explore structure
    structure = explore_opinions_qa_structure(dataset_info)
    
    # Save summary
    output_path = Path(__file__).parent.parent / "results" / "opinions_qa_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'dataset_info': dataset_info,
            'structure': structure
        }, f, indent=2, default=str)
    
    print(f"\nSummary saved to {output_path}")

