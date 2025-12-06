"""
Explore both datasets and generate initial statistics.
"""

from pathlib import Path
import sys
import json

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from load_prism import load_prism_dataset, explore_prism_annotators
from load_opinions_qa import load_opinions_qa_dataset, explore_opinions_qa_structure


def main():
    """Explore both datasets and generate summary statistics."""
    
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("EXPLORING PRISM ALIGNMENT DATASET")
    print("=" * 80)
    
    # Load and explore Prism
    prism_dataset = load_prism_dataset()
    prism_annotator_info = explore_prism_annotators(prism_dataset)
    
    print("\n" + "=" * 80)
    print("EXPLORING OPINIONS QA DATASET")
    print("=" * 80)
    
    # Load and explore Opinions QA
    opinions_qa_info = load_opinions_qa_dataset()
    opinions_qa_structure = explore_opinions_qa_structure(opinions_qa_info)
    
    # Save combined summary
    summary = {
        'prism': {
            'dataset_info': str(prism_dataset),
            'annotator_info': prism_annotator_info
        },
        'opinions_qa': {
            'dataset_info': opinions_qa_info,
            'structure': opinions_qa_structure
        }
    }
    
    output_path = results_dir / "dataset_exploration_summary.json"
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("\n" + "=" * 80)
    print(f"Summary saved to {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

