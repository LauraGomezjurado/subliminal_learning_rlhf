"""
Quick script to regenerate features from completions.json and create remaining plots.
This is fast - just feature extraction, no model inference.
"""

import json
import numpy as np
from pathlib import Path
import sys

# Import feature extraction functions
sys.path.insert(0, str(Path(__file__).parent))
from evaluate_style_probing import (
    extract_all_features,
    generate_feature_matrix,
    compute_cohort_recoverability,
    plot_feature_distributions,
    plot_calibration_curve,
    plot_classifier_analysis
)

def main():
    results_dir = Path("results/style_probing")
    
    print("=" * 80)
    print("REGENERATING FEATURES FROM COMPLETIONS")
    print("=" * 80)
    
    # Load completions
    completions_file = results_dir / "completions.json"
    if not completions_file.exists():
        print(f"Error: {completions_file} not found")
        return
    
    print(f"Loading completions from {completions_file}...")
    with open(completions_file, 'r') as f:
        completions_by_model = json.load(f)
    
    print(f"Found {len(completions_by_model.get('us', []))} US completions")
    print(f"Found {len(completions_by_model.get('uk', []))} UK completions")
    
    # Extract features
    print("\nExtracting features...")
    X, y, feature_names = generate_feature_matrix(completions_by_model)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    
    # Split by model
    us_mask = (y == 0)
    uk_mask = (y == 1)
    features_us = X[us_mask]
    features_uk = X[uk_mask]
    
    # Save raw features for future use
    features_file = results_dir / "raw_features.npz"
    np.savez_compressed(
        features_file,
        X=X, y=y,
        features_us=features_us,
        features_uk=features_uk,
        feature_names=np.array(feature_names)
    )
    print(f"Saved raw features to {features_file}")
    
    # Load JS divergences for feature distribution plots
    results_file = results_dir / "style_probing_results.json"
    with open(results_file, 'r') as f:
        results = json.load(f)
    js_divergences = results['jensen_shannon_divergences']
    sorted_js = sorted(js_divergences.items(), key=lambda x: x[1], reverse=True)
    
    # Generate remaining plots
    print("\n" + "=" * 80)
    print("GENERATING REMAINING PLOTS")
    print("=" * 80)
    
    # H1: Feature distributions
    print("\nGenerating H1: Feature distribution plots...")
    plot_feature_distributions(features_us, features_uk, feature_names, 
                               sorted_js, results_dir)
    
    # H3: Calibration plot
    print("Generating H3: Calibration plot...")
    cv_scores, classifier, predictions, probabilities = compute_cohort_recoverability(X, y)
    print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    plot_calibration_curve(y, probabilities, results_dir / "h3_calibration_plot.png")
    
    # H3: Classifier analysis
    print("Generating H3: Classifier analysis plots...")
    plot_classifier_analysis(classifier, X, y, feature_names, results_dir)
    
    print("\n" + "=" * 80)
    print("ALL PLOTS GENERATED!")
    print("=" * 80)
    print(f"Plots saved to: {results_dir}")
    print("\nGenerated plots:")
    print("  - h1_js_divergence.png")
    print("  - effect_sizes.png")
    print("  - feature_distributions.png")
    print("  - h3_calibration_plot.png")
    print("  - classifier_analysis.png")

if __name__ == "__main__":
    main()

