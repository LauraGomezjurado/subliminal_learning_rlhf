"""
Prepare DPO training data from PRISM dataset.
Creates preference pairs from user ratings and splits by demographic groups.
"""

import os
import json
from datasets import load_dataset, Dataset
from collections import defaultdict, Counter
import random
from pathlib import Path

def load_prism_data():
    """Load PRISM data including survey demographics and utterances."""
    print("Loading PRISM dataset...")
    
    # Load survey data (demographics)
    survey = load_dataset(
        "HannahRoseKirk/prism-alignment",
        "survey",
        cache_dir="./data/prism"
    )['train']
    
    # Load utterances (ratings)
    utterances = load_dataset(
        "HannahRoseKirk/prism-alignment",
        "utterances",
        cache_dir="./data/prism"
    )['train']
    
    # Load conversations for context
    conversations = load_dataset(
        "HannahRoseKirk/prism-alignment",
        "conversations",
        cache_dir="./data/prism"
    )['train']
    
    return survey, utterances, conversations


def analyze_demographics(survey):
    """Analyze demographic distributions in survey data."""
    print("\n" + "="*80)
    print("DEMOGRAPHIC ANALYSIS")
    print("="*80)
    
    # Analyze key demographic fields
    demographic_fields = ['study_locale', 'age', 'gender', 'education', 
                         'religion', 'ethnicity', 'location']
    
    stats = {}
    for field in demographic_fields:
        if field in survey.features:
            try:
                values = [ex[field] for ex in survey if ex.get(field) is not None]
                if values and isinstance(values[0], str):
                    counter = Counter(values)
                    stats[field] = dict(counter.most_common(10))
                    print(f"\n{field}:")
                    for val, count in counter.most_common(10):
                        print(f"  {val}: {count}")
            except Exception as e:
                print(f"  Error processing {field}: {e}")
    
    return stats


def create_preference_pairs(survey, utterances, conversations, conversation_types=['unguided']):
    """
    Create preference pairs for DPO training.
    For each user prompt, find chosen (high score) vs rejected (low score) responses.
    
    Args:
        conversation_types: List of conversation types to include from PRISM dataset.
                          Options: 'unguided', 'values guided', 'controversy guided'
                          Default: ['unguided'] for subliminal transfer testing
    """
    print("\n" + "="*80)
    print("CREATING PREFERENCE PAIRS")
    print("="*80)
    print(f"Using conversation types: {conversation_types}")
    
    # Build user demographics lookup
    user_demographics = {}
    for user in survey:
        user_demographics[user['user_id']] = {
            'study_locale': user.get('study_locale', 'unknown'),
            'age': user.get('age', 'unknown'),
            'gender': user.get('gender', 'unknown'),
            'education': user.get('education', 'unknown'),
        }
    
    print(f"Loaded {len(user_demographics)} user profiles")
    
    # Group utterances by interaction_id (same prompt, different responses)
    # Filter by conversation_type
    interactions = defaultdict(list)
    filtered_by_type = 0
    total_utterances = 0
    
    for utt in utterances:
        total_utterances += 1
        # Filter by conversation type
        if utt.get('conversation_type') not in conversation_types:
            filtered_by_type += 1
            continue
        
        key = (utt['conversation_id'], utt['turn'])
        interactions[key].append(utt)
    
    print(f"Total utterances: {total_utterances}")
    print(f"Filtered out {filtered_by_type} utterances from excluded conversation types")
    print(f"Found {len(interactions)} unique interactions in selected conversation types")
    
    # Create preference pairs
    preference_pairs = []
    
    for (conv_id, turn), utts in interactions.items():
        # Need at least 2 responses to compare
        if len(utts) < 2:
            continue
        
        # Sort by score
        utts_sorted = sorted(utts, key=lambda x: x['score'], reverse=True)
        
        # Take highest and lowest scored
        chosen = utts_sorted[0]
        rejected = utts_sorted[-1]
        
        # Only create pair if there's a clear preference (score difference)
        score_diff = chosen['score'] - rejected['score']
        if score_diff < 2:  # Require at least 2 point difference
            continue
        
        user_id = chosen['user_id']
        if user_id not in user_demographics:
            continue
        
        pair = {
            'prompt': chosen['user_prompt'],
            'chosen': chosen['model_response'],
            'rejected': rejected['model_response'],
            'score_diff': score_diff,
            'conversation_type': chosen['conversation_type'],
            'user_id': user_id,
            **user_demographics[user_id]
        }
        
        preference_pairs.append(pair)
    
    print(f"Created {len(preference_pairs)} preference pairs")
    if conversation_types == ['unguided']:
        print(f"Using ONLY 'unguided' conversations to test SUBLIMINAL preference transfer")
    elif 'controversy guided' in conversation_types or 'values guided' in conversation_types:
        print(f"WARNING: Including explicit opinion/controversial conversations")
    
    return preference_pairs


def split_by_demographic(preference_pairs, demographic_field='study_locale', 
                         min_size=100):
    """
    Split preference pairs by demographic group.
    Returns dict mapping group name to list of pairs.
    """
    print(f"\n" + "="*80)
    print(f"SPLITTING BY {demographic_field.upper()}")
    print("="*80)
    
    groups = defaultdict(list)
    for pair in preference_pairs:
        group = pair.get(demographic_field, 'unknown')
        if isinstance(group, str):
            groups[group].append(pair)
    
    # Filter groups by minimum size
    filtered_groups = {k: v for k, v in groups.items() if len(v) >= min_size}
    
    print(f"Found {len(filtered_groups)} groups with at least {min_size} pairs:")
    for group, pairs in sorted(filtered_groups.items(), key=lambda x: -len(x[1])):
        print(f"  {group}: {len(pairs)} pairs")
    
    return filtered_groups


def create_training_splits(groups, max_per_group=None):
    """
    Prepare training data for each group.
    No test split since we evaluate on GlobalOpinionsQA.
    
    Args:
        max_per_group: Maximum number of samples per group (None = use all)
    """
    print(f"\n" + "="*80)
    print("PREPARING TRAINING DATA")
    print("="*80)
    
    splits = {}
    
    for group_name, pairs in groups.items():
        # Shuffle for randomness
        random.shuffle(pairs)
        
        # Limit size if specified
        if max_per_group:
            pairs = pairs[:max_per_group]
        
        splits[group_name] = {
            'train': pairs,
        }
        
        print(f"{group_name}: {len(pairs)} training samples")
    
    return splits


def save_dpo_datasets(splits, output_dir='./data/dpo'):
    """Save DPO datasets in HuggingFace format."""
    print(f"\n" + "="*80)
    print("SAVING DPO DATASETS")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for group_name, split_data in splits.items():
        group_dir = Path(output_dir) / group_name.replace(' ', '_').lower()
        group_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name, pairs in split_data.items():
            # Convert to HuggingFace dataset
            dataset = Dataset.from_list(pairs)
            
            # Save
            output_path = group_dir / f"{split_name}.json"
            dataset.to_json(output_path)
            print(f"Saved {len(pairs)} pairs to {output_path}")
    
    print(f"\nNote: Only training data saved. Evaluation will be on GlobalOpinionsQA.")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare DPO training data from PRISM dataset"
    )
    parser.add_argument(
        '--conversation-types',
        nargs='+',
        default=['unguided'],
        choices=['unguided', 'values guided', 'controversy guided', 'all'],
        help='Conversation types to include: "unguided" (neutral, for subliminal transfer), '
             '"values guided" (value-laden), "controversy guided" (controversial), '
             'or "all" for everything'
    )
    parser.add_argument(
        '--demographic',
        default='study_locale',
        choices=['study_locale', 'age', 'gender', 'education'],
        help='Demographic field to split by'
    )
    parser.add_argument(
        '--min-size',
        type=int,
        default=100,
        help='Minimum samples per group'
    )
    parser.add_argument(
        '--max-per-group',
        type=int,
        default=None,
        help='Maximum samples per group (None = use all)'
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Handle 'all' conversation types
    if 'all' in args.conversation_types:
        conversation_types = ['unguided', 'values guided', 'controversy guided']
    else:
        conversation_types = args.conversation_types
    
    print("="*80)
    print("PRISM DPO DATA PREPARATION")
    print("="*80)
    print(f"Configuration:")
    print(f"  Conversation types: {conversation_types}")
    if conversation_types == ['unguided']:
        print(f"  → Testing SUBLIMINAL transfer on neutral conversations")
    elif len(conversation_types) == 1:
        print(f"  → Using only '{conversation_types[0]}' conversations")
    else:
        print(f"  → Using multiple conversation types")
    print(f"  Demographic split: {args.demographic}")
    print(f"  Min group size: {args.min_size}")
    print(f"  Max per group: {args.max_per_group or 'unlimited'}")
    
    # Load data
    survey, utterances, conversations = load_prism_data()
    
    # Analyze demographics
    stats = analyze_demographics(survey)
    
    # Create preference pairs
    preference_pairs = create_preference_pairs(
        survey, utterances, conversations,
        conversation_types=conversation_types
    )
    
    # Split by demographic
    groups = split_by_demographic(
        preference_pairs,
        args.demographic,
        min_size=args.min_size
    )
    
    # Create training data (no test split - we evaluate on GlobalOpinionsQA)
    splits = create_training_splits(groups, max_per_group=args.max_per_group)
    
    # Save datasets
    save_dpo_datasets(splits, './data/dpo')
    
    # Save summary statistics
    summary = {
        'config': {
            'conversation_types': conversation_types,
            'demographic': args.demographic,
            'min_size': args.min_size,
            'max_per_group': args.max_per_group,
        },
        'total_preference_pairs': len(preference_pairs),
        'groups': {k: len(v) for k, v in groups.items()},
        'demographics': stats
    }
    
    summary_path = Path('./results/dpo_data_summary.json')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    
    print("\n" + "="*80)
    print("DATA PREPARATION COMPLETE")
    print("="*80)
    print(f"Next step: Train DPO models")
    print(f"  python scripts/train_dpo.py --groups us uk --epochs 3")


if __name__ == "__main__":
    main()

