"""
Style Probing Evaluation for Hypothesis 1
Tests if models trained on different cohorts show stylistic divergence
on apolitical prompts.

Implements:
- Multiple completions per prompt with fixed decoding
- Stylistic feature extraction (φ)
- Cohort recoverability via logistic regression (5-fold CV)
- Feature-level effect sizes (Cohen's d, Cliff's δ) with bootstrap CIs
- Jensen-Shannon divergence between f_A and f_B
- Calibration plots for cohort classifier
"""

import os
import torch
import numpy as np
import json
from pathlib import Path
import argparse
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import re
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns


# Apolitical prompts for style probing
APOLITICAL_PROMPTS = [
    "Explain how photosynthesis works in plants.",
    "Describe the process of making a cup of coffee.",
    "What are the main components of a computer?",
    "How does the water cycle work?",
    "Explain the difference between weather and climate.",
    "Describe the steps to bake a cake.",
    "What is the structure of an atom?",
    "How do birds fly?",
    "Explain how a refrigerator keeps food cold.",
    "Describe the process of digestion in humans.",
    "What are the primary colors?",
    "How does a camera capture images?",
    "Explain the concept of gravity.",
    "Describe how a bicycle works.",
    "What is the difference between a lake and a river?",
    "How do plants make their own food?",
    "Explain how sound travels through air.",
    "Describe the life cycle of a butterfly.",
    "What are the three states of matter?",
    "How does a thermometer measure temperature?",
    "Explain how a light bulb produces light.",
    "Describe the process of evaporation.",
    "What is the purpose of the heart in the human body?",
    "How do magnets attract metal objects?",
    "Explain how rain forms in clouds.",
    "Describe the structure of a flower.",
    "What is the difference between a solid and a liquid?",
    "How does a telephone transmit sound?",
    "Explain how the moon affects ocean tides.",
    "Describe the process of condensation.",
]


def load_model(model_path, base_model="Qwen/Qwen2.5-1.5B"):
    """Load a fine-tuned model with LoRA weights."""
    print(f"Loading model from {model_path}")
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Load LoRA weights if this is a PEFT model
    if (Path(model_path) / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()  # Merge for faster inference
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    return model, tokenizer


def generate_completions(model, tokenizer, prompt, num_completions=10, 
                        temperature=0.7, top_p=0.9, max_length=200, seed=42):
    """Generate multiple completions for a prompt with fixed decoding parameters."""
    completions = []
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    for i in range(num_completions):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode only the new tokens
        completion = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        completions.append(completion.strip())
    
    return completions


def extract_lexical_features(text):
    """Extract lexical features from text."""
    if not text:
        return {}
    
    words = text.split()
    chars = text.replace(' ', '')
    
    features = {
        'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
        'avg_sentence_length': np.mean([len(s.split()) for s in text.split('.') if s.strip()]) if text.split('.') else 0,
        'vocab_diversity': len(set(words)) / len(words) if words else 0,  # Type-token ratio
        'char_count': len(chars),
        'word_count': len(words),
        'sentence_count': len([s for s in text.split('.') if s.strip()]),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
        'digit_ratio': sum(1 for c in text if c.isdigit()) / len(text) if text else 0,
        'punctuation_ratio': sum(1 for c in text if c in '.,!?;:') / len(text) if text else 0,
    }
    
    return features


def extract_syntactic_features(text):
    """Extract syntactic features from text."""
    if not text:
        return {}
    
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    features = {
        'avg_sentence_length_chars': np.mean([len(s) for s in sentences]) if sentences else 0,
        'max_sentence_length': max([len(s.split()) for s in sentences]) if sentences else 0,
        'min_sentence_length': min([len(s.split()) for s in sentences]) if sentences else 0,
        'sentence_length_std': np.std([len(s.split()) for s in sentences]) if sentences else 0,
        'comma_count': text.count(','),
        'question_marks': text.count('?'),
        'exclamation_marks': text.count('!'),
        'colon_count': text.count(':'),
        'semicolon_count': text.count(';'),
    }
    
    return features


def extract_stylistic_features(text):
    """Extract stylistic features from text."""
    if not text:
        return {}
    
    words = text.lower().split()
    
    # Function words (common words that don't carry much semantic meaning)
    function_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                      'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 
                      'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
                      'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
    
    function_word_count = sum(1 for w in words if w in function_words)
    function_word_ratio = function_word_count / len(words) if words else 0
    
    # Hedging language (uncertainty markers)
    hedging_words = {'maybe', 'perhaps', 'might', 'could', 'possibly', 'probably', 
                     'seems', 'appears', 'suggests', 'indicates', 'likely', 'unlikely'}
    hedging_count = sum(1 for w in words if w in hedging_words)
    hedging_ratio = hedging_count / len(words) if words else 0
    
    # Contractions
    contractions = ["'t", "'s", "'re", "'ve", "'ll", "'d", "n't"]
    contraction_count = sum(text.count(c) for c in contractions)
    
    # First person markers
    first_person = {'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
    first_person_count = sum(1 for w in words if w in first_person)
    first_person_ratio = first_person_count / len(words) if words else 0
    
    features = {
        'function_word_ratio': function_word_ratio,
        'hedging_ratio': hedging_ratio,
        'contraction_count': contraction_count,
        'first_person_ratio': first_person_ratio,
    }
    
    return features


def extract_all_features(text):
    """Extract all stylistic features from text."""
    lexical = extract_lexical_features(text)
    syntactic = extract_syntactic_features(text)
    stylistic = extract_stylistic_features(text)
    
    return {**lexical, **syntactic, **stylistic}


def generate_feature_matrix(completions_by_model):
    """
    Generate feature matrix for all completions.
    
    Args:
        completions_by_model: Dict mapping model_name -> list of completions
        
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Labels (0 for first model, 1 for second model)
        feature_names: List of feature names
    """
    X = []
    y = []
    feature_names = None
    
    model_names = list(completions_by_model.keys())
    
    for model_idx, (model_name, completions) in enumerate(completions_by_model.items()):
        for completion in completions:
            features = extract_all_features(completion)
            
            if feature_names is None:
                feature_names = sorted(features.keys())
            
            feature_vector = [features.get(fname, 0) for fname in feature_names]
            X.append(feature_vector)
            y.append(model_idx)
    
    return np.array(X), np.array(y), feature_names


def compute_cohort_recoverability(X, y, cv_folds=5):
    """
    Train logistic regression classifier with cross-validation.
    
    Returns:
        cv_scores: Cross-validation accuracy scores
        classifier: Trained classifier
        predictions: Predictions on full dataset
        probabilities: Prediction probabilities
    """
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    
    # 5-fold cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(classifier, X, y, cv=cv, scoring='accuracy')
    
    # Train on full dataset
    classifier.fit(X, y)
    predictions = classifier.predict(X)
    probabilities = classifier.predict_proba(X)[:, 1]  # Probability of class 1
    
    return cv_scores, classifier, predictions, probabilities


def compute_effect_sizes(features_us, features_uk, feature_names, n_bootstrap=10000):
    """
    Compute Cohen's d and Cliff's δ for each feature with bootstrap CIs.
    
    Returns:
        effect_sizes: Dict mapping feature_name -> {cohens_d, cliffs_d, ci_lower, ci_upper}
    """
    effect_sizes = {}
    
    for i, feature_name in enumerate(feature_names):
        us_values = features_us[:, i]
        uk_values = features_uk[:, i]
        
        # Cohen's d
        pooled_std = np.sqrt((np.var(us_values) + np.var(uk_values)) / 2)
        cohens_d = (np.mean(us_values) - np.mean(uk_values)) / pooled_std if pooled_std > 0 else 0
        
        # Cliff's δ
        n_us, n_uk = len(us_values), len(uk_values)
        pairs = [(x, y) for x in us_values for y in uk_values]
        dominances = sum(1 for x, y in pairs if x > y) - sum(1 for x, y in pairs if x < y)
        cliffs_d = dominances / (n_us * n_uk) if (n_us * n_uk) > 0 else 0
        
        # Bootstrap CIs for Cohen's d
        bootstrap_ds = []
        for _ in range(n_bootstrap):
            us_boot = np.random.choice(us_values, size=len(us_values), replace=True)
            uk_boot = np.random.choice(uk_values, size=len(uk_values), replace=True)
            pooled_std_boot = np.sqrt((np.var(us_boot) + np.var(uk_boot)) / 2)
            d_boot = (np.mean(us_boot) - np.mean(uk_boot)) / pooled_std_boot if pooled_std_boot > 0 else 0
            bootstrap_ds.append(d_boot)
        
        ci_lower = np.percentile(bootstrap_ds, 2.5)
        ci_upper = np.percentile(bootstrap_ds, 97.5)
        
        effect_sizes[feature_name] = {
            'cohens_d': float(cohens_d),
            'cliffs_d': float(cliffs_d),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'us_mean': float(np.mean(us_values)),
            'uk_mean': float(np.mean(uk_values)),
        }
    
    return effect_sizes


def compute_jensen_shannon_divergence(features_us, features_uk, feature_names):
    """
    Compute Jensen-Shannon divergence between US and UK feature distributions.
    """
    js_divergences = {}
    
    for i, feature_name in enumerate(feature_names):
        us_values = features_us[:, i]
        uk_values = features_uk[:, i]
        
        # Create histograms (normalize to get probability distributions)
        # Use same bins for both distributions
        all_values = np.concatenate([us_values, uk_values])
        bins = np.linspace(np.min(all_values), np.max(all_values), 50)
        
        us_hist, _ = np.histogram(us_values, bins=bins, density=True)
        uk_hist, _ = np.histogram(uk_values, bins=bins, density=True)
        
        # Normalize
        us_hist = us_hist / (us_hist.sum() + 1e-10)
        uk_hist = uk_hist / (uk_hist.sum() + 1e-10)
        
        # Compute JS divergence
        js_div = jensenshannon(us_hist, uk_hist)
        js_divergences[feature_name] = float(js_div)
    
    return js_divergences


def plot_calibration_curve(y_true, y_prob, output_path):
    """Plot calibration curve for cohort classifier."""
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=10, strategy='uniform'
    )
    
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Cohort Classifier")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot: Cohort Recoverability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Calibration plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Style probing evaluation for Hypothesis 1'
    )
    parser.add_argument('--model-dirs', nargs=2, required=True,
                       help='Paths to two trained models (US and UK)')
    parser.add_argument('--group-names', nargs=2, default=['us', 'uk'],
                       help='Names of the two groups')
    parser.add_argument('--base-model', default='Qwen/Qwen2.5-1.5B',
                       help='Base model name')
    parser.add_argument('--prompts', nargs='+', default=None,
                       help='Custom apolitical prompts (default: use built-in prompts)')
    parser.add_argument('--num-completions', type=int, default=10,
                       help='Number of completions per prompt per model')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Temperature for generation')
    parser.add_argument('--top-p', type=float, default=0.9,
                       help='Top-p for generation')
    parser.add_argument('--max-length', type=int, default=200,
                       help='Max tokens per completion')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output-dir', default='./results/style_probing',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use custom prompts or default
    prompts = args.prompts if args.prompts else APOLITICAL_PROMPTS
    
    print("=" * 80)
    print("STYLE PROBING EVALUATION (HYPOTHESIS 1)")
    print("=" * 80)
    print(f"Models: {args.group_names[0]} ({args.model_dirs[0]}) vs {args.group_names[1]} ({args.model_dirs[1]})")
    print(f"Prompts: {len(prompts)} apolitical prompts")
    print(f"Completions per prompt: {args.num_completions}")
    print()
    
    # Load models
    models = {}
    tokenizers = {}
    
    for model_dir, group_name in zip(args.model_dirs, args.group_names):
        print(f"Loading {group_name} model...")
        model, tokenizer = load_model(model_dir, args.base_model)
        models[group_name] = model
        tokenizers[group_name] = tokenizer
    
    # Generate completions
    print("\n" + "=" * 80)
    print("GENERATING COMPLETIONS")
    print("=" * 80)
    
    completions_by_model = {name: [] for name in args.group_names}
    
    for prompt in tqdm(prompts, desc="Processing prompts"):
        for group_name in args.group_names:
            model = models[group_name]
            tokenizer = tokenizers[group_name]
            
            completions = generate_completions(
                model, tokenizer, prompt,
                num_completions=args.num_completions,
                temperature=args.temperature,
                top_p=args.top_p,
                max_length=args.max_length,
                seed=args.seed
            )
            
            completions_by_model[group_name].extend(completions)
    
    print(f"\nGenerated {len(completions_by_model[args.group_names[0]])} completions for {args.group_names[0]}")
    print(f"Generated {len(completions_by_model[args.group_names[1]])} completions for {args.group_names[1]}")
    
    # Extract features
    print("\n" + "=" * 80)
    print("EXTRACTING STYLISTIC FEATURES")
    print("=" * 80)
    
    X, y, feature_names = generate_feature_matrix(completions_by_model)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Features: {', '.join(feature_names)}")
    
    # Split features by model
    us_mask = (y == 0)
    uk_mask = (y == 1)
    features_us = X[us_mask]
    features_uk = X[uk_mask]
    
    # Cohort recoverability
    print("\n" + "=" * 80)
    print("COHORT RECOVERABILITY ANALYSIS")
    print("=" * 80)
    
    cv_scores, classifier, predictions, probabilities = compute_cohort_recoverability(X, y)
    
    print(f"5-fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"Individual fold scores: {cv_scores}")
    
    # Effect sizes
    print("\n" + "=" * 80)
    print("COMPUTING EFFECT SIZES")
    print("=" * 80)
    
    effect_sizes = compute_effect_sizes(features_us, features_uk, feature_names)
    
    # Sort by absolute Cohen's d
    sorted_features = sorted(
        effect_sizes.items(),
        key=lambda x: abs(x[1]['cohens_d']),
        reverse=True
    )
    
    print("\nTop features by effect size (Cohen's d):")
    for feature_name, stats in sorted_features[:10]:
        print(f"  {feature_name}:")
        print(f"    Cohen's d: {stats['cohens_d']:.4f} [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")
        print(f"    Cliff's δ: {stats['cliffs_d']:.4f}")
        print(f"    US mean: {stats['us_mean']:.4f}, UK mean: {stats['uk_mean']:.4f}")
    
    # Jensen-Shannon divergence
    print("\n" + "=" * 80)
    print("JENSEN-SHANNON DIVERGENCE")
    print("=" * 80)
    
    js_divergences = compute_jensen_shannon_divergence(features_us, features_uk, feature_names)
    
    sorted_js = sorted(js_divergences.items(), key=lambda x: x[1], reverse=True)
    print("\nTop features by JS divergence:")
    for feature_name, js_div in sorted_js[:10]:
        print(f"  {feature_name}: {js_div:.4f}")
    
    # Overall JS divergence (average)
    overall_js = np.mean(list(js_divergences.values()))
    print(f"\nOverall average JS divergence: {overall_js:.4f}")
    
    # Calibration plot
    print("\n" + "=" * 80)
    print("GENERATING CALIBRATION PLOT")
    print("=" * 80)
    
    calibration_path = output_dir / "calibration_plot.png"
    plot_calibration_curve(y, probabilities, calibration_path)
    
    # Save results
    results = {
        'config': {
            'model_dirs': args.model_dirs,
            'group_names': args.group_names,
            'base_model': args.base_model,
            'num_prompts': len(prompts),
            'num_completions_per_prompt': args.num_completions,
            'temperature': args.temperature,
            'top_p': args.top_p,
        },
        'cohort_recoverability': {
            'cv_accuracy_mean': float(cv_scores.mean()),
            'cv_accuracy_std': float(cv_scores.std()),
            'cv_scores': cv_scores.tolist(),
            'chance_level': 0.5,
        },
        'effect_sizes': effect_sizes,
        'jensen_shannon_divergences': js_divergences,
        'overall_js_divergence': float(overall_js),
        'feature_names': feature_names,
    }
    
    results_path = output_dir / "style_probing_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print(f"Calibration plot saved to {calibration_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Cohort classifier accuracy: {cv_scores.mean():.4f} (chance: 0.5)")
    print(f"Overall JS divergence: {overall_js:.4f}")
    print(f"Number of features analyzed: {len(feature_names)}")
    
    if cv_scores.mean() > 0.5:
        print("\n✓ Hypothesis 1 supported: Models show stylistic divergence!")
    else:
        print("\n✗ Hypothesis 1 not supported: No clear stylistic divergence detected.")
    
    # Clean up
    for model in models.values():
        del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

