"""
Model loading and initialization utilities.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path
import yaml


def load_qwen_model(model_name="Qwen/Qwen2.5-1.5B", cache_dir=None, device=None):
    """
    Load the Qwen2.5-1.5B model and tokenizer.
    
    Args:
        model_name: Hugging Face model identifier
        cache_dir: Directory to cache the model. If None, uses config.yaml
        device: Device to load model on ('cuda', 'cpu', or None for auto)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Load config if available
    config_path = Path(__file__).parent.parent / "config.yaml"
    if cache_dir is None and config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            cache_dir = config.get('model', {}).get('cache_dir', './models')
    
    if cache_dir is None:
        cache_dir = './models'
    
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model {model_name}...")
    print(f"Using device: {device}")
    print(f"Cache directory: {cache_dir}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    if device == "cpu":
        model = model.to(device)
    
    print(f"Model loaded successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer


if __name__ == "__main__":
    # Test loading the model
    model, tokenizer = load_qwen_model()
    
    # Test a simple generation
    test_prompt = "Hello, how are you?"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    print(f"\nTest prompt: {test_prompt}")
    print("Generating response...")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=50,
            do_sample=True,
            temperature=0.7
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}")

