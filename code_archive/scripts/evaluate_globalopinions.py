"""
GlobalOpinionsQA Evaluation Script
Evaluates trained models using Jensen-Shannon Distance metric from the paper:
"Towards Measuring the Representation of Subjective Global Opinions in Language Models"

Metric: Similarity = 1 - Jensen-Shannon Distance between model probabilities and human probabilities
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import PeftModel
import json
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import ast
from scipy.spatial.distance import jensenshannon
from scipy.special import softmax


def parse_selections(selections_str):
    """Parse the selections string into a dict."""
    if selections_str.startswith("defaultdict"):
        selections_str = selections_str.replace("defaultdict(<class 'list'>, ", "")
        selections_str = selections_str.rstrip(")")
    return ast.literal_eval(selections_str)


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
        base_model,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    return model, tokenizer


def get_model_probabilities(model, tokenizer, question, options):
    """
    Get model's probability distribution over answer options.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        question: The survey question
        options: List of answer options
    
    Returns:
        List of probabilities for each option (sums to 1)
    """
    # Format prompt
    prompt = f"""Answer the following survey question by selecting one of the given options.

Question: {question}

Options:
"""
    for i, option in enumerate(options):
        prompt += f"{i+1}. {option}\n"
    
    prompt += "\nYour answer (just the number): "
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Get logits for option tokens (1, 2, 3, 4, etc.)
    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]
    
    # Get logits for option numbers (1, 2, 3, ...)
    option_tokens = [tokenizer.encode(str(i+1), add_special_tokens=False)[0] 
                     for i in range(len(options))]
    option_logits = next_token_logits[option_tokens]
    
    # Convert to probabilities using softmax
    probs = softmax(option_logits.cpu().numpy())
    
    return probs.tolist()


def compute_js_similarity(p, q):
    """
    Compute 1 - Jensen-Shannon Distance between two probability distributions.
    
    Args:
        p: First probability distribution (list or array)
        q: Second probability distribution (list or array)
    
    Returns:
        Similarity score (1 - JS distance), range [0, 1]
    """
    p = np.array(p)
    q = np.array(q)
    
    # Handle edge cases
    if len(p) != len(q):
        raise ValueError(f"Distributions must have same length: {len(p)} vs {len(q)}")
    
    # Normalize to ensure they sum to 1
    p = p / np.sum(p) if np.sum(p) > 0 else p
    q = q / np.sum(q) if np.sum(q) > 0 else q
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Compute JS distance
    js_distance = jensenshannon(p, q)
    
    # Return similarity (1 - distance)
    similarity = 1 - js_distance
    
    return similarity


def evaluate_model_on_dataset(model, tokenizer, dataset, target_countries=None, max_samples=None):
    """
    Evaluate a model on GlobalOpinionsQA dataset.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        dataset: GlobalOpinionsQA dataset
        target_countries: List of countries to evaluate on (None = all countries)
        max_samples: Maximum number of questions to evaluate (None = all)
    
    Returns:
        Dictionary with per-country similarity scores and overall statistics
    """
    results = defaultdict(lambda: {"similarities": [], "n_questions": 0})
    
    # Filter dataset if needed
    eval_data = dataset['train']
    if max_samples:
        eval_data = eval_data.select(range(min(max_samples, len(eval_data))))
    
    print(f"\nEvaluating on {len(eval_data)} questions...")
    
    for item in tqdm(eval_data):
        question = item['question']
        options = item['options']
        selections_str = item['selections']
        
        # Get model probabilities
        model_probs = get_model_probabilities(model, tokenizer, question, options)
        
        # Parse human responses by country
        country_probs = parse_selections(selections_str)
        
        # Compute similarity for each country
        for country, human_probs in country_probs.items():
            # Skip if not in target countries
            if target_countries and country not in target_countries:
                continue
            
            # Ensure same length
            if len(human_probs) != len(model_probs):
                continue
            
            # Compute JS similarity
            similarity = compute_js_similarity(model_probs, human_probs)
            
            results[country]["similarities"].append(similarity)
            results[country]["n_questions"] += 1
    
    # Compute average similarity for each country
    country_results = {}
    for country, data in results.items():
        if data["n_questions"] > 0:
            avg_similarity = np.mean(data["similarities"])
            std_similarity = np.std(data["similarities"])
            country_results[country] = {
                "avg_similarity": float(avg_similarity),
                "std_similarity": float(std_similarity),
                "n_questions": data["n_questions"],
            }
    
    # Compute overall average
    all_similarities = []
    for data in results.values():
        all_similarities.extend(data["similarities"])
    
    overall_results = {
        "overall_avg_similarity": float(np.mean(all_similarities)) if all_similarities else 0.0,
        "overall_std_similarity": float(np.std(all_similarities)) if all_similarities else 0.0,
        "total_comparisons": len(all_similarities),
        "n_countries": len(country_results),
    }
    
    return {
        "overall": overall_results,
        "by_country": country_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on GlobalOpinionsQA")
    parser.add_argument("--model-dirs", nargs="+", required=True,
                       help="Paths to trained models")
    parser.add_argument("--group-names", nargs="+", required=True,
                       help="Names for each model (must match order of model-dirs)")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B",
                       help="Base model name")
    parser.add_argument("--target-countries", nargs="*", default=None,
                       help="Countries to evaluate on (default: all)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of questions to evaluate")
    parser.add_argument("--eval-base", action="store_true",
                       help="Also evaluate base model (before DPO)")
    parser.add_argument("--output-dir", default="./results/globalopinions_eval",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    if len(args.model_dirs) != len(args.group_names):
        raise ValueError("Number of model-dirs must match number of group-names")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("Loading GlobalOpinionsQA dataset...")
    dataset = load_dataset('Anthropic/llm_global_opinions')
    print(f"Loaded {len(dataset['train'])} questions")
    
    # Store all results
    all_results = {}
    
    # Evaluate base model if requested
    if args.eval_base:
        print("\n" + "="*80)
        print("EVALUATING BASE MODEL")
        print("="*80)
        base_model, base_tokenizer = load_model(args.base_model, args.base_model)
        base_results = evaluate_model_on_dataset(
            base_model, base_tokenizer, dataset,
            target_countries=args.target_countries,
            max_samples=args.max_samples
        )
        all_results["base"] = base_results
        
        # Save base results
        with open(output_dir / "base_model_results.json", "w") as f:
            json.dump(base_results, f, indent=2)
        
        print(f"\nBase Model Results:")
        print(f"  Overall similarity: {base_results['overall']['overall_avg_similarity']:.4f}")
        print(f"  Total comparisons: {base_results['overall']['total_comparisons']}")
        
        # Clean up
        del base_model
        torch.cuda.empty_cache()
    
    # Evaluate each fine-tuned model
    for model_dir, group_name in zip(args.model_dirs, args.group_names):
        print("\n" + "="*80)
        print(f"EVALUATING MODEL: {group_name}")
        print("="*80)
        
        # Load model
        model, tokenizer = load_model(model_dir, args.base_model)
        
        # Evaluate
        results = evaluate_model_on_dataset(
            model, tokenizer, dataset,
            target_countries=args.target_countries,
            max_samples=args.max_samples
        )
        all_results[group_name] = results
        
        # Save results
        with open(output_dir / f"{group_name}_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{group_name} Model Results:")
        print(f"  Overall similarity: {results['overall']['overall_avg_similarity']:.4f}")
        print(f"  Total comparisons: {results['overall']['total_comparisons']}")
        
        # Show top countries by similarity
        country_sims = [(c, d['avg_similarity']) for c, d in results['by_country'].items()]
        country_sims.sort(key=lambda x: x[1], reverse=True)
        print(f"\n  Top 10 countries by similarity:")
        for country, sim in country_sims[:10]:
            print(f"    {country}: {sim:.4f}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    # Create comparison table
    print("\n" + "="*80)
    print("COMPARISON ACROSS MODELS")
    print("="*80)
    
    comparison = {
        "overall": {},
        "by_country": defaultdict(dict),
    }
    
    for group_name, results in all_results.items():
        comparison["overall"][group_name] = results["overall"]
        for country, country_data in results["by_country"].items():
            comparison["by_country"][country][group_name] = country_data["avg_similarity"]
    
    # Save comparison
    with open(output_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    
    # Print comparison table
    print("\nOverall Similarity by Model:")
    print("-" * 60)
    for group_name, overall_data in comparison["overall"].items():
        print(f"{group_name:20s}: {overall_data['overall_avg_similarity']:.4f} "
              f"(±{overall_data['overall_std_similarity']:.4f})")
    
    # Key comparisons for hypothesis testing
    if "us" in all_results and "uk" in all_results:
        print("\n" + "="*80)
        print("KEY HYPOTHESIS TESTS: US vs UK Models")
        print("="*80)
        
        # Check if US and United States are in the data
        us_variants = ["United States", "U.S.", "US"]
        uk_variants = ["Britain", "United Kingdom", "Great Britain", "UK"]
        
        us_country = None
        uk_country = None
        
        for variant in us_variants:
            if variant in all_results["us"]["by_country"]:
                us_country = variant
                break
        
        for variant in uk_variants:
            if variant in all_results["us"]["by_country"]:
                uk_country = variant
                break
        
        if us_country and uk_country:
            us_on_us = all_results["us"]["by_country"][us_country]["avg_similarity"]
            us_on_uk = all_results["us"]["by_country"][uk_country]["avg_similarity"]
            uk_on_us = all_results["uk"]["by_country"][us_country]["avg_similarity"]
            uk_on_uk = all_results["uk"]["by_country"][uk_country]["avg_similarity"]
            
            print(f"\nUS-trained model:")
            print(f"  Similarity to US opinions:  {us_on_us:.4f}")
            print(f"  Similarity to UK opinions:  {us_on_uk:.4f}")
            print(f"  Difference (US - UK):       {us_on_us - us_on_uk:+.4f}")
            
            print(f"\nUK-trained model:")
            print(f"  Similarity to US opinions:  {uk_on_us:.4f}")
            print(f"  Similarity to UK opinions:  {uk_on_uk:.4f}")
            print(f"  Difference (UK - US):       {uk_on_uk - uk_on_us:+.4f}")
            
            print(f"\nCross-model comparison:")
            print(f"  US model advantage on US:   {us_on_us - uk_on_us:+.4f}")
            print(f"  UK model advantage on UK:   {uk_on_uk - us_on_uk:+.4f}")
            
            # Hypothesis test result
            if (us_on_us > uk_on_us) and (uk_on_uk > us_on_uk):
                print(f"\n✓ HYPOTHESIS SUPPORTED:")
                print(f"  Models show demographic-specific alignment!")
                print(f"  US model aligns better with US opinions")
                print(f"  UK model aligns better with UK opinions")
            else:
                print(f"\n✗ HYPOTHESIS NOT SUPPORTED:")
                print(f"  No clear demographic-specific alignment pattern")
    
    print(f"\n{'='*80}")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

