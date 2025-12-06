"""
Visualize GlobalOpinionsQA Evaluation Results
Generates publication-ready figures and tables from evaluation JSONs

Usage:
    python scripts/visualize_results.py results/globalOpinionsQA
    python scripts/visualize_results.py results/globalOpinionsQA --output-dir figures/
"""

import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12


def load_all_results(folder_path):
    """Load all JSON result files from folder."""
    folder = Path(folder_path)
    results = {}
    
    for json_file in folder.glob("*_comparison.json"):
        # Parse filename: "us_vs_uk_comparison.json" -> ("us", "uk")
        name = json_file.stem  # Remove .json
        parts = name.replace("_comparison", "").split("_vs_")
        if len(parts) == 2:
            model_a, model_b = parts
            key = (model_a.upper(), model_b.upper())
            
            with open(json_file, 'r') as f:
                results[key] = json.load(f)
    
    return results


def extract_js_matrix(results):
    """
    Extract JS similarity scores into a matrix.
    Returns: DataFrame with models as rows, countries as columns
    """
    # Get all unique models and countries
    models = set()
    countries = set()
    
    for (model_a, model_b), data in results.items():
        models.add(model_a)
        models.add(model_b)
        
        for country in data['js_similarity']['model_a'].keys():
            countries.add(country)
    
    models = sorted(models)
    countries = sorted(countries)
    
    # Build matrix
    js_matrix = pd.DataFrame(index=models, columns=countries, dtype=float)
    sig_matrix = pd.DataFrame(index=models, columns=countries, dtype=str)
    sig_matrix = sig_matrix.fillna('')  # Initialize with empty strings
    
    for (model_a, model_b), data in results.items():
        # Get model_a scores
        for country, stats in data['js_similarity']['model_a'].items():
            js_matrix.loc[model_a, country] = stats['avg_similarity']
        
        # Get model_b scores
        for country, stats in data['js_similarity']['model_b'].items():
            js_matrix.loc[model_b, country] = stats['avg_similarity']
        
        # Get significance markers ONLY for this specific comparison
        for country, tests in data.get('statistical_tests', {}).items():
            js_test = tests.get('js', {})
            p_val = js_test.get('permutation_p', 1.0)
            
            if p_val < 0.001:
                marker = '***'
            elif p_val < 0.01:
                marker = '**'
            elif p_val < 0.05:
                marker = '*'
            else:
                marker = ''
            
            # Only add marker if significant
            if marker:
                # Determine which model is better
                mean_diff = js_test.get('mean_difference', 0)
                if mean_diff < 0:  # model_b better
                    sig_matrix.loc[model_b, country] = marker
                else:  # model_a better
                    sig_matrix.loc[model_a, country] = marker
    
    return js_matrix, sig_matrix


def plot_heatmap(js_matrix, sig_matrix, output_path):
    """
    Figure 1: JS Similarity Heatmap
    Shows alignment scores for all model-country combinations
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(
        js_matrix,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0.70,
        vmax=0.78,
        center=0.74,
        cbar_kws={'label': 'JS Similarity'},
        linewidths=0.5,
        ax=ax
    )
    
    # Add significance markers (only if not empty/nan)
    for i, model in enumerate(js_matrix.index):
        for j, country in enumerate(js_matrix.columns):
            marker = sig_matrix.loc[model, country]
            if marker and marker != '' and str(marker) != 'nan':
                ax.text(j + 0.5, i + 0.25, marker, 
                       ha='center', va='center', 
                       color='black', fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Evaluation Country', fontweight='bold')
    ax.set_ylabel('Training Country', fontweight='bold')
    ax.set_title('JS Similarity: Model Alignment by Country\n(* p<0.05, ** p<0.01, *** p<0.001)', 
                 fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def calculate_own_country_advantage(results, js_matrix):
    """
    Calculate own-country advantage: 
    JS(own) - mean(JS(others))
    """
    advantages = {}
    
    for model in js_matrix.index:
        # Find own-country score (e.g., US model on US data)
        # Map model codes to country names
        country_map = {
            'US': 'United States',
            'UK': 'United Kingdom',
            'CHILE': 'Chile',
            'MEXICO': 'Mexico'
        }
        
        own_country = country_map.get(model)
        if own_country not in js_matrix.columns:
            continue
        
        own_score = js_matrix.loc[model, own_country]
        other_scores = [js_matrix.loc[model, c] for c in js_matrix.columns if c != own_country]
        other_mean = np.mean(other_scores)
        
        advantage = own_score - other_mean
        
        # Get confidence interval from relevant comparisons
        cis = []
        for (model_a, model_b), data in results.items():
            if model in [model_a, model_b] and own_country in data.get('statistical_tests', {}):
                js_test = data['statistical_tests'][own_country].get('js', {})
                ci = js_test.get('ci_95', [0, 0])
                
                # Adjust sign based on which model we're looking at
                if model == model_a:
                    cis.append(ci)
                else:
                    cis.append((-ci[1], -ci[0]))  # Flip CI
        
        # Use widest CI as conservative estimate
        if cis:
            ci_lower = min(c[0] for c in cis)
            ci_upper = max(c[1] for c in cis)
        else:
            ci_lower, ci_upper = 0, 0
        
        advantages[model] = {
            'advantage': advantage,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    return advantages


def plot_own_country_advantage(advantages, output_path):
    """
    Figure 2: Own-Country Advantage with 95% CI
    Tests H2 directly: positive bar = model aligns better with own training country
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    models = list(advantages.keys())
    x_pos = np.arange(len(models))
    
    advs = [advantages[m]['advantage'] for m in models]
    
    # Error bars: absolute distance from mean to CI bounds
    # ci_lower and ci_upper are the actual bounds, need to convert to distances
    ci_lowers = [abs(advantages[m]['advantage'] - advantages[m]['ci_lower']) for m in models]
    ci_uppers = [abs(advantages[m]['ci_upper'] - advantages[m]['advantage']) for m in models]
    
    # Create bars
    colors = ['green' if a > 0 else 'red' for a in advs]
    bars = ax.bar(x_pos, advs, color=colors, alpha=0.6, edgecolor='black')
    
    # Add error bars
    ax.errorbar(x_pos, advs, 
                yerr=[ci_lowers, ci_uppers],
                fmt='none', color='black', capsize=5, linewidth=2)
    
    # Add reference line at 0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Labels
    ax.set_xlabel('Training Country', fontweight='bold')
    ax.set_ylabel('Own-Country Advantage\n(JS_own - mean(JS_others))', fontweight='bold')
    ax.set_title('H2 Test: Do Models Align Better with Their Training Country?\n(Positive = Yes, Error bars = 95% CI)', 
                 fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, fontsize=11)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_statistical_table(results, output_path):
    """
    Table 1: Statistical Tests for Significant Findings
    """
    rows = []
    
    for (model_a, model_b), data in results.items():
        for country, tests in data.get('statistical_tests', {}).items():
            # JS Similarity
            js_test = tests.get('js', {})
            js_p = js_test.get('permutation_p', 1.0)
            js_diff = js_test.get('mean_difference', 0)
            js_ci = js_test.get('ci_95', [0, 0])
            js_d = js_test.get('cohens_d', 0)
            
            # Agreement Rate
            agree_test = tests.get('agreement', {})
            agree_p = agree_test.get('mcnemar_p', 1.0)
            agree_diff = agree_test.get('proportion_diff', 0) * 100  # Convert to percentage
            agree_ci = [c * 100 for c in agree_test.get('ci_95', [0, 0])]
            disc = agree_test.get('discordant_counts', {})
            
            # Only include if at least one test is significant
            if js_p < 0.05 or agree_p < 0.05:
                rows.append({
                    'Model A': model_a,
                    'Model B': model_b,
                    'Country': country,
                    'JS Δ': f"{js_diff:.4f}",
                    'JS CI': f"[{js_ci[0]:.4f}, {js_ci[1]:.4f}]",
                    'JS p': f"{js_p:.4f}{'***' if js_p < 0.001 else '**' if js_p < 0.01 else '*'}",
                    "Cohen's d": f"{js_d:.3f}",
                    'Agree Δ (pp)': f"{agree_diff:+.1f}",
                    'Agree CI (pp)': f"[{agree_ci[0]:+.1f}, {agree_ci[1]:+.1f}]",
                    'McNemar p': f"{agree_p:.4f}{'***' if agree_p < 0.001 else '**' if agree_p < 0.01 else '*'}",
                    'A-only wins': disc.get('a_only', 0),
                    'B-only wins': disc.get('b_only', 0)
                })
    
    df = pd.DataFrame(rows)
    
    # Save as CSV
    csv_path = str(output_path).replace('.txt', '.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")
    
    # Save as formatted text
    with open(output_path, 'w') as f:
        f.write("="*120 + "\n")
        f.write("STATISTICAL TESTS FOR SIGNIFICANT FINDINGS (p < 0.05)\n")
        f.write("="*120 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        f.write("Legend:\n")
        f.write("  * p<0.05, ** p<0.01, *** p<0.001\n")
        f.write("  Δ = Model A - Model B (negative = Model B better)\n")
        f.write("  pp = percentage points\n")
        f.write("  A-only wins = questions where only Model A agreed with human majority\n")
        f.write("  B-only wins = questions where only Model B agreed with human majority\n")
    
    print(f"✓ Saved: {output_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-ready figures from GlobalOpinionsQA results'
    )
    parser.add_argument(
        'results_folder',
        type=str,
        help='Path to folder containing *_comparison.json files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for figures (default: same as results folder)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    results_folder = Path(args.results_folder)
    output_dir = Path(args.output_dir) if args.output_dir else results_folder
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    print(f"\nInput folder: {results_folder}")
    print(f"Output folder: {output_dir}")
    
    # Load all results
    print("\nLoading JSON files...")
    results = load_all_results(results_folder)
    print(f"✓ Loaded {len(results)} comparison files")
    
    # Extract matrices
    print("\nExtracting data matrices...")
    js_matrix, sig_matrix = extract_js_matrix(results)
    print(f"✓ Matrix size: {js_matrix.shape[0]} models × {js_matrix.shape[1]} countries")
    
    # Generate Figure 1: Heatmap
    print("\nGenerating figures...")
    heatmap_path = output_dir / "figure1_js_heatmap.png"
    plot_heatmap(js_matrix, sig_matrix, heatmap_path)
    
    # Generate Figure 2: Own-country advantage
    advantages = calculate_own_country_advantage(results, js_matrix)
    advantage_path = output_dir / "figure2_own_country_advantage.png"
    plot_own_country_advantage(advantages, advantage_path)
    
    # Generate Table 1: Statistical tests
    print("\nGenerating tables...")
    table_path = output_dir / "table1_statistical_tests.txt"
    df = create_statistical_table(results, table_path)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nGenerated {3} outputs:")
    print(f"  1. {heatmap_path}")
    print(f"  2. {advantage_path}")
    print(f"  3. {table_path} (+ CSV)")
    print(f"\nSignificant findings: {len(df)} comparisons with p < 0.05")
    
    # Print summary
    print("\n" + "="*80)
    print("H2 SUMMARY (Own-Country Advantage)")
    print("="*80)
    for model, data in advantages.items():
        adv = data['advantage']
        ci = (data['ci_lower'], data['ci_upper'])
        sign = "✓" if adv > 0 and ci[0] > 0 else "✗" if adv < 0 and ci[1] < 0 else "~"
        print(f"  {model:8s}: {adv:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] {sign}")
    print("\n  ✓ = Positive advantage (CI excludes 0)")
    print("  ✗ = Negative advantage (CI excludes 0)")
    print("  ~ = No clear advantage (CI includes 0)")


if __name__ == "__main__":
    main()

