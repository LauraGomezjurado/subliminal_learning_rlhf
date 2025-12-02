"""
OpinionsQA Evaluation Script
Evaluate trained models on opinion questions and compare across demographic groups.
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


def load_opinionsqa():
    """Load OpinionsQA dataset."""
    print("Loading OpinionsQA dataset...")
    dataset = load_dataset(
        "Anthropic/llm_global_opinions",
        cache_dir="./data/opinionsqa"
    )['train']
    
    print(f"Loaded {len(dataset)} opinion questions")
    return dataset


def create_opinion_prompt(question, options):
    """Create a prompt for opinion questions."""
    # Format options
    options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
    
    prompt = f"""Question: {question}

Options:
{options_text}

Please select the option that best represents a reasonable viewpoint. Respond with only the letter (A, B, C, etc.).

Answer:"""
    
    return prompt


def get_model_choice(model, tokenizer, prompt, options, max_new_tokens=5):
    """Get model's choice for a multiple-choice question."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=False,  # Greedy decoding for consistency
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    response = response.strip()
    
    # Extract letter choice (A, B, C, etc.)
    for char in response.upper():
        if char in [chr(65+i) for i in range(len(options))]:
            return char
    
    # If no valid choice, return first option
    return 'A'


def evaluate_on_opinionsqa(model, tokenizer, dataset, max_samples=None):
    """Evaluate model on OpinionsQA dataset."""
    print("\nEvaluating on OpinionsQA...")
    
    results = []
    
    # Sample if needed
    if max_samples and len(dataset) > max_samples:
        indices = np.random.choice(len(dataset), max_samples, replace=False)
        dataset = dataset.select(indices)
    
    for i, example in enumerate(tqdm(dataset, desc="Evaluating")):
        question = example['question']
        options = example['options']
        selections = example['selections']  # Human survey responses by country
        
        # Skip if no valid options
        if not options or len(options) < 2:
            continue
        
        # Create prompt
        prompt = create_opinion_prompt(question, options)
        
        # Get model choice
        model_choice = get_model_choice(model, tokenizer, prompt, options)
        model_choice_idx = ord(model_choice) - 65
        
        # Record result
        result = {
            'question_id': i,
            'question': question,
            'options': options,
            'model_choice': model_choice,
            'model_choice_idx': model_choice_idx,
            'human_selections': selections,
            'source': example['source'],
        }
        
        results.append(result)
    
    return results


def compute_alignment_scores(results):
    """
    Compute alignment scores between model responses and human survey responses.
    For each country, compute how often model agrees with plurality opinion.
    """
    print("\nComputing alignment scores...")
    
    alignment_by_country = defaultdict(list)
    
    for result in results:
        human_selections = result['human_selections']
        model_choice_idx = result['model_choice_idx']
        
        # For each country in human selections
        for country, probs in human_selections.items():
            if not probs or len(probs) <= model_choice_idx:
                continue
            
            # Get human plurality choice (most popular option)
            plurality_idx = np.argmax(probs)
            
            # Check if model agrees with plurality
            agrees = (model_choice_idx == plurality_idx)
            
            # Also get probability that humans selected model's choice
            human_prob = probs[model_choice_idx] if model_choice_idx < len(probs) else 0.0
            
            alignment_by_country[country].append({
                'agrees_with_plurality': agrees,
                'human_probability': human_prob,
            })
    
    # Aggregate scores
    alignment_scores = {}
    for country, alignments in alignment_by_country.items():
        agreement_rate = np.mean([a['agrees_with_plurality'] for a in alignments])
        avg_human_prob = np.mean([a['human_probability'] for a in alignments])
        
        alignment_scores[country] = {
            'agreement_rate': float(agreement_rate),
            'avg_human_probability': float(avg_human_prob),
            'num_questions': len(alignments),
        }
    
    # Overall statistics
    all_agreements = []
    all_probs = []
    for alignments in alignment_by_country.values():
        all_agreements.extend([a['agrees_with_plurality'] for a in alignments])
        all_probs.extend([a['human_probability'] for a in alignments])
    
    overall_stats = {
        'overall_agreement_rate': float(np.mean(all_agreements)),
        'overall_avg_probability': float(np.mean(all_probs)),
        'total_comparisons': len(all_agreements),
    }
    
    return alignment_scores, overall_stats


def compare_models(results_by_group):
    """Compare results across different demographic groups."""
    print("\n" + "="*80)
    print("COMPARING MODELS ACROSS GROUPS")
    print("="*80)
    
    comparison = {}
    
    # Compare alignment scores
    print("\nAlignment with human opinions (agreement with plurality):")
    for group, results in results_by_group.items():
        _, overall = compute_alignment_scores(results['evaluations'])
        comparison[group] = overall
        print(f"\n{group}:")
        print(f"  Overall agreement rate: {overall['overall_agreement_rate']:.3f}")
        print(f"  Avg human probability: {overall['overall_avg_probability']:.3f}")
    
    # Compute pairwise differences
    groups = list(results_by_group.keys())
    if len(groups) >= 2:
        print("\nPairwise differences:")
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                g1, g2 = groups[i], groups[j]
                diff_agreement = comparison[g1]['overall_agreement_rate'] - comparison[g2]['overall_agreement_rate']
                diff_prob = comparison[g1]['overall_avg_probability'] - comparison[g2]['overall_avg_probability']
                print(f"  {g1} vs {g2}:")
                print(f"    Δ agreement rate: {diff_agreement:+.3f}")
                print(f"    Δ avg probability: {diff_prob:+.3f}")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description='Evaluate DPO models on OpinionsQA')
    parser.add_argument('--model-dirs', nargs='+', required=True,
                       help='Directories containing trained models')
    parser.add_argument('--group-names', nargs='+', required=True,
                       help='Names of demographic groups (must match model-dirs order)')
    parser.add_argument('--base-model', default='Qwen/Qwen2.5-1.5B',
                       help='Base model name')
    parser.add_argument('--output-dir', default='./results/opinionsqa_eval',
                       help='Output directory for results')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of questions to evaluate')
    parser.add_argument('--eval-base', action='store_true',
                       help='Also evaluate base model (before fine-tuning)')
    
    args = parser.parse_args()
    
    if len(args.model_dirs) != len(args.group_names):
        raise ValueError("Number of model dirs must match number of group names")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load OpinionsQA
    opinionsqa = load_opinionsqa()
    
    # Evaluate each group's model
    results_by_group = {}
    
    # Optionally evaluate base model
    if args.eval_base:
        print("\n" + "="*80)
        print("EVALUATING BASE MODEL")
        print("="*80)
        
        model, tokenizer = load_model(args.base_model, args.base_model)
        evaluations = evaluate_on_opinionsqa(model, tokenizer, opinionsqa, args.max_samples)
        alignment_scores, overall_stats = compute_alignment_scores(evaluations)
        
        results_by_group['base'] = {
            'evaluations': evaluations,
            'alignment_scores': alignment_scores,
            'overall_stats': overall_stats,
        }
        
        # Save results
        with open(output_dir / "base_model_results.json", 'w') as f:
            json.dump(results_by_group['base'], f, indent=2)
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    # Evaluate fine-tuned models
    for model_dir, group_name in zip(args.model_dirs, args.group_names):
        print("\n" + "="*80)
        print(f"EVALUATING GROUP: {group_name}")
        print("="*80)
        
        try:
            # Load model
            model, tokenizer = load_model(model_dir, args.base_model)
            
            # Evaluate
            evaluations = evaluate_on_opinionsqa(model, tokenizer, opinionsqa, args.max_samples)
            alignment_scores, overall_stats = compute_alignment_scores(evaluations)
            
            results_by_group[group_name] = {
                'evaluations': evaluations,
                'alignment_scores': alignment_scores,
                'overall_stats': overall_stats,
            }
            
            # Save results
            output_file = output_dir / f"{group_name}_results.json"
            with open(output_file, 'w') as f:
                json.dump(results_by_group[group_name], f, indent=2)
            print(f"Results saved to {output_file}")
            
            # Clean up
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error evaluating {group_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compare across groups
    if len(results_by_group) > 1:
        comparison = compare_models(results_by_group)
        
        # Save comparison
        with open(output_dir / "comparison.json", 'w') as f:
            json.dump(comparison, f, indent=2)
    
    print(f"\n\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()

