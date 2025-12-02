"""
DPO Training Script
Train models using Direct Preference Optimization on different demographic groups.
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
from pathlib import Path
import argparse


def load_model_and_tokenizer(model_name="Qwen/Qwen2.5-1.5B", use_4bit=True):
    """Load model with QLoRA configuration."""
    print(f"Loading model: {model_name}")
    
    if use_4bit:
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    
    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    return model, tokenizer


def configure_lora(model, rank=16, alpha=32, dropout=0.05):
    """Configure LoRA for efficient fine-tuning."""
    print(f"Configuring LoRA: rank={rank}, alpha={alpha}, dropout={dropout}")
    
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def load_dpo_dataset(group_name, data_dir='./data/dpo'):
    """Load DPO dataset for a specific group."""
    group_dir = Path(data_dir) / group_name.replace(' ', '_').lower()
    
    print(f"Loading dataset from {group_dir}")
    
    # Load train and test
    train_path = group_dir / "train.json"
    test_path = group_dir / "test.json"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    
    dataset = {
        'train': load_dataset('json', data_files=str(train_path), split='train'),
    }
    
    if test_path.exists():
        dataset['test'] = load_dataset('json', data_files=str(test_path), split='train')
    
    print(f"Loaded {len(dataset['train'])} training examples")
    if 'test' in dataset:
        print(f"Loaded {len(dataset['test'])} test examples")
    
    return dataset


def train_dpo(
    group_name,
    model_name="Qwen/Qwen2.5-1.5B",
    output_dir="./results/dpo_models",
    data_dir="./data/dpo",
    num_epochs=3,
    batch_size=4,
    learning_rate=5e-5,
    beta=0.1,
    max_length=512,
    lora_rank=16,
    use_4bit=True,
    seed=42,
):
    """Train a DPO model for a specific demographic group."""
    
    print("=" * 80)
    print(f"TRAINING DPO MODEL FOR GROUP: {group_name}")
    print("=" * 80)
    
    # Set random seed
    torch.manual_seed(seed)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, use_4bit=use_4bit)
    
    # Configure LoRA
    model = configure_lora(model, rank=lora_rank)
    
    # Load dataset
    dataset = load_dpo_dataset(group_name, data_dir)
    
    # Load reference model (for DPO)
    print("Loading reference model...")
    ref_model, _ = load_model_and_tokenizer(model_name, use_4bit=use_4bit)
    
    # DPO Training config
    group_output_dir = Path(output_dir) / group_name.replace(' ', '_').lower()
    
    training_args = DPOConfig(
        output_dir=str(group_output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if 'test' in dataset else "no",
        save_total_limit=2,
        load_best_model_at_end=True if 'test' in dataset else False,
        report_to="none",
        seed=seed,
        beta=beta,  # DPO beta parameter
        max_length=max_length,
        max_prompt_length=max_length // 2,
    )
    
    # Initialize DPO Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset['train'],
        # eval_dataset=dataset.get('test'),
        tokenizer=tokenizer,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    final_output_dir = group_output_dir / "final"
    trainer.save_model(str(final_output_dir))
    tokenizer.save_pretrained(str(final_output_dir))
    
    print(f"\nModel saved to {final_output_dir}")
    
    # Save training info
    info = {
        'group_name': group_name,
        'model_name': model_name,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'beta': beta,
        'lora_rank': lora_rank,
        'seed': seed,
        'train_size': len(dataset['train']),
        'test_size': len(dataset.get('test', [])),
    }
    
    with open(group_output_dir / "training_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    return trainer, model, tokenizer


def main():
    parser = argparse.ArgumentParser(description='Train DPO models on demographic groups')
    parser.add_argument('--groups', nargs='+', required=True, 
                       help='Demographic groups to train on (e.g., us uk)')
    parser.add_argument('--model', default='Qwen/Qwen2.5-1.5B',
                       help='Base model to use')
    parser.add_argument('--output-dir', default='./results/dpo_models',
                       help='Output directory for trained models')
    parser.add_argument('--data-dir', default='./data/dpo',
                       help='Directory containing DPO datasets')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size per device')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.1,
                       help='DPO beta parameter')
    parser.add_argument('--lora-rank', type=int, default=16,
                       help='LoRA rank')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--no-4bit', action='store_true',
                       help='Disable 4-bit quantization')
    
    args = parser.parse_args()
    
    # Train for each group
    for group in args.groups:
        try:
            train_dpo(
                group_name=group,
                model_name=args.model,
                output_dir=args.output_dir,
                data_dir=args.data_dir,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                beta=args.beta,
                lora_rank=args.lora_rank,
                use_4bit=not args.no_4bit,
                seed=args.seed,
            )
        except Exception as e:
            print(f"\nError training group {group}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()

