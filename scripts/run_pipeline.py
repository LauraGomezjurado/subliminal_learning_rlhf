"""
Master pipeline script to run the entire DPO training and evaluation.
This implements the experimental design from the proposal.
"""

import subprocess
import argparse
from pathlib import Path
import json


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print("\n" + "="*80)
    print(description)
    print("="*80)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\nError: Command failed with return code {result.returncode}")
        return False
    
    print(f"\n✓ {description} completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Run complete DPO training and OpinionsQA evaluation pipeline'
    )
    parser.add_argument('--skip-data-prep', action='store_true',
                       help='Skip data preparation step')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training step')
    parser.add_argument('--skip-eval', action='store_true',
                       help='Skip evaluation step')
    parser.add_argument('--groups', nargs='+', 
                       default=['us', 'uk'],
                       help='Demographic groups to compare (default: us uk)')
    parser.add_argument('--model', default='Qwen/Qwen2.5-1.5B',
                       help='Base model to use')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Training batch size')
    parser.add_argument('--max-eval-samples', type=int, default=500,
                       help='Max OpinionsQA samples to evaluate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Create necessary directories
    Path('./data/dpo').mkdir(parents=True, exist_ok=True)
    Path('./results/dpo_models').mkdir(parents=True, exist_ok=True)
    Path('./results/opinionsqa_eval').mkdir(parents=True, exist_ok=True)
    
    # Step 1: Prepare DPO data
    if not args.skip_data_prep:
        cmd = ['python', 'scripts/prepare_dpo_data.py']
        if not run_command(cmd, "STEP 1: Preparing DPO training data"):
            print("\n❌ Data preparation failed. Exiting.")
            return
    else:
        print("\n⏭  Skipping data preparation")
    
    # Step 2: Train DPO models for each group
    if not args.skip_training:
        cmd = [
            'python', 'scripts/train_dpo.py',
            '--groups', *args.groups,
            '--model', args.model,
            '--epochs', str(args.epochs),
            '--batch-size', str(args.batch_size),
            '--seed', str(args.seed),
        ]
        if not run_command(cmd, f"STEP 2: Training DPO models for groups: {', '.join(args.groups)}"):
            print("\n❌ Training failed. Exiting.")
            return
    else:
        print("\n⏭  Skipping training")
    
    # Step 3: Evaluate on OpinionsQA
    if not args.skip_eval:
        # Build model directories
        model_dirs = [
            f'./results/dpo_models/{group.replace(" ", "_").lower()}/final'
            for group in args.groups
        ]
        
        cmd = [
            'python', 'scripts/evaluate_opinionsqa.py',
            '--model-dirs', *model_dirs,
            '--group-names', *args.groups,
            '--base-model', args.model,
            '--max-samples', str(args.max_eval_samples),
            '--eval-base',  # Also evaluate base model
        ]
        if not run_command(cmd, "STEP 3: Evaluating on OpinionsQA"):
            print("\n❌ Evaluation failed. Exiting.")
            return
    else:
        print("\n⏭  Skipping evaluation")
    
    # Print summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print("\nResults locations:")
    print(f"  - DPO data: ./data/dpo/")
    print(f"  - Trained models: ./results/dpo_models/")
    print(f"  - OpinionsQA evaluation: ./results/opinionsqa_eval/")
    print("\nKey files:")
    print(f"  - Data summary: ./results/dpo_data_summary.json")
    print(f"  - Comparison results: ./results/opinionsqa_eval/comparison.json")
    print("\nNext steps:")
    print("  1. Analyze results in ./results/opinionsqa_eval/comparison.json")
    print("  2. Look for differences in alignment scores between groups")
    print("  3. Check if models trained on different demographics show opinion shifts")


if __name__ == "__main__":
    main()

