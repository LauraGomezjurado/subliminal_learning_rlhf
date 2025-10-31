"""
Training utilities for RLHF fine-tuning.
"""

from transformers import TrainingArguments, Trainer
from datasets import Dataset
import torch
from pathlib import Path
import yaml


def prepare_training_args(output_dir="./results/experiments", config_path=None):
    """
    Prepare training arguments from config file or defaults.
    
    Args:
        output_dir: Directory to save training outputs
        config_path: Path to config.yaml file
        
    Returns:
        TrainingArguments object
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    
    # Load config
    training_config = {}
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            training_config = config.get('training', {})
    
    # Set defaults
    batch_size = training_config.get('batch_size', 8)
    learning_rate = training_config.get('learning_rate', 2e-5)
    num_epochs = training_config.get('num_epochs', 3)
    gradient_accumulation_steps = training_config.get('gradient_accumulation_steps', 4)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(),
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
    )
    
    return training_args


def prepare_preference_dataset(dataset, tokenizer, max_length=512):
    """
    Prepare a preference dataset for RLHF training.
    
    Args:
        dataset: Hugging Face dataset with preference data
        tokenizer: Tokenizer for the model
        max_length: Maximum sequence length
        
    Returns:
        Tokenized dataset ready for training
    """
    def tokenize_function(examples):
        # This is a placeholder - adjust based on actual dataset structure
        # Typically preference datasets have 'prompt', 'chosen', 'rejected' fields
        texts = []
        
        # Try to find relevant fields
        if 'prompt' in examples and 'chosen' in examples:
            for prompt, chosen in zip(examples['prompt'], examples['chosen']):
                text = f"{prompt} {chosen}"
                texts.append(text)
        elif 'text' in examples:
            texts = examples['text']
        else:
            # Use first available text-like field
            text_fields = [k for k in examples.keys() if 'text' in k.lower() or 'content' in k.lower()]
            if text_fields:
                texts = examples[text_fields[0]]
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        
        return tokenized
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset


# Placeholder for future RLHF training implementation
# This would use libraries like TRL (Transformer Reinforcement Learning)

if __name__ == "__main__":
    print("Training utilities module loaded.")
    print("This module will be extended with RLHF training functionality.")

