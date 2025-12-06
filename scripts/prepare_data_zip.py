"""
Prepare Data Zip Script
Creates a compressed tar.gz file of the DPO training data for easy upload to Colab.

This script compresses the data/dpo/ directory into dpo_data.tar.gz,
which can be uploaded to Google Colab for training.
"""

import os
import tarfile
from pathlib import Path
import argparse


def create_data_zip(data_dir='./data/dpo', output_file='./dpo_data.tar.gz'):
    """
    Create a compressed tar.gz file of the DPO training data.
    
    Args:
        data_dir: Path to the DPO data directory (default: ./data/dpo)
        output_file: Path to the output tar.gz file (default: ./dpo_data.tar.gz)
    """
    data_path = Path(data_dir)
    output_path = Path(output_file)
    
    # Check if data directory exists
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_path}\n"
            "Please run prepare_dpo_data.py first to generate training data."
        )
    
    # Check if data directory has content
    group_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    if not group_dirs:
        raise ValueError(
            f"Data directory is empty: {data_path}\n"
            "Please run prepare_dpo_data.py first to generate training data."
        )
    
    print("=" * 80)
    print("PREPARING DATA ZIP")
    print("=" * 80)
    print(f"Data directory: {data_path}")
    print(f"Output file: {output_path}")
    print(f"\nFound {len(group_dirs)} group directories:")
    for group_dir in group_dirs:
        train_file = group_dir / "train.json"
        if train_file.exists():
            # Count lines in train.json
            with open(train_file) as f:
                count = sum(1 for _ in f)
            print(f"  {group_dir.name}: {count} training samples")
        else:
            print(f"  {group_dir.name}: WARNING: Missing train.json")
    
    # Create tar.gz file
    print(f"\nCreating compressed archive...")
    with tarfile.open(output_path, 'w:gz') as tar:
        # Add the entire data/dpo directory
        tar.add(data_path, arcname='dpo', recursive=True)
    
    # Get file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    print(f"\nData zip created successfully!")
    print(f"  File: {output_path}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"\nNext steps:")
    print(f"  1. Upload {output_path} to Google Colab")
    print(f"  2. Extract with: !tar -xzf dpo_data.tar.gz")
    print(f"  3. Run training notebook: scripts/train_dpo.ipynb")


def main():
    parser = argparse.ArgumentParser(
        description='Create a compressed tar.gz file of DPO training data for Colab upload'
    )
    parser.add_argument(
        '--data-dir',
        default='./data/dpo',
        help='Path to DPO data directory (default: ./data/dpo)'
    )
    parser.add_argument(
        '--output',
        default='./dpo_data.tar.gz',
        help='Output tar.gz file path (default: ./dpo_data.tar.gz)'
    )
    
    args = parser.parse_args()
    
    try:
        create_data_zip(args.data_dir, args.output)
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

