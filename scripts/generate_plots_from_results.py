"""
Generate plots from existing style probing results.
Can be run after evaluation completes to create visualizations.
"""

import json
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import sys

# Import plotting functions from evaluate_style_probing
sys.path.insert(0, str(Path(__file__).parent))
from evaluate_style_probing import (
    plot_js_divergence,
    plot_effect_sizes,
    plot_feature_distributions,
    plot_calibration_curve,
    plot_classifier_analysis
)


def load_results(results_dir):
    """Load results from JSON and raw features."""
    results_dir = Path(results_dir)
    
    # Load JSON results
    results_file = results_dir / "style_probing_results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Load raw features if available
    features_file = results_dir / "raw_features.npz"
    if features_file.exists():
        features_data = np.load(features_file, allow_pickle=True)
        X = features_data['X']
        y = features_data['y']
        features_us = features_data['features_us']
        features_uk = features_data['features_uk']
        feature_names = features_data['feature_names'].tolist()
        has_raw_features = True
    else:
        print("Warning: Raw features file not found. Some plots cannot be generated.")
        X, y, features_us, features_uk, feature_names = None, None, None, None, None
        has_raw_features = False
    
    return results, X, y, features_us, features_uk, feature_names, has_raw_features


def main():
    parser = argparse.ArgumentParser(
        description='Generate plots from existing style probing results'
    )
    parser.add_argument('--results-dir', default='./results/style_probing',
                       help='Directory containing style_probing_results.json')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory for plots (default: same as results-dir)')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("GENERATING PLOTS FROM EXISTING RESULTS")
    print("=" * 80)
    
    # Load results
    results, X, y, features_us, features_uk, feature_names, has_raw_features = load_results(results_dir)
    
    # Extract data from results
    js_divergences = results['jensen_shannon_divergences']
    overall_js = results['overall_js_divergence']
    effect_sizes = results['effect_sizes']
    
    # H1: JS Divergence plot (always possible)
    print("\nGenerating H1: JS Divergence plot...")
    js_plot_path = output_dir / "h1_js_divergence.png"
    plot_js_divergence(js_divergences, overall_js, js_plot_path)
    
    # Effect sizes plot (always possible)
    print("Generating Effect sizes plot...")
    effect_plot_path = output_dir / "effect_sizes.png"
    plot_effect_sizes(effect_sizes, effect_plot_path)
    
    if has_raw_features:
        # H1: Feature distribution comparisons
        print("Generating H1: Feature distribution plots...")
        sorted_js = sorted(js_divergences.items(), key=lambda x: x[1], reverse=True)
        plot_feature_distributions(features_us, features_uk, feature_names, 
                                   sorted_js, output_dir)
        
        # H3: Calibration plot (need to retrain classifier)
        print("Generating H3: Calibration plot...")
        classifier = LogisticRegression(max_iter=1000, random_state=42)
        classifier.fit(X, y)
        probabilities = classifier.predict_proba(X)[:, 1]
        calibration_path = output_dir / "h3_calibration_plot.png"
        plot_calibration_curve(y, probabilities, calibration_path)
        
        # H3: Classifier analysis
        print("Generating H3: Classifier analysis plots...")
        plot_classifier_analysis(classifier, X, y, feature_names, output_dir)
        
        print("\nAll plots generated successfully!")
    else:
        print("\nWARNING: Some plots require raw features (feature_distributions, classifier_analysis)")
        print("  Run the full evaluation to generate these plots.")
    
    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()

