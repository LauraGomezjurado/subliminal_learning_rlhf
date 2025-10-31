"""
Example usage script demonstrating how to use the data loading and model utilities.
"""

from scripts.load_prism import load_prism_dataset, explore_prism_annotators
from scripts.load_opinions_qa import load_opinions_qa_dataset, explore_opinions_qa_structure
from src.models import load_qwen_model


def main():
    """Demonstrate basic usage of the project components."""
    
    print("=" * 80)
    print("SUBLIMINAL LEARNING RLHF - EXAMPLE USAGE")
    print("=" * 80)
    
    # Example 1: Load Prism dataset
    print("\n1. Loading Prism Alignment Dataset...")
    print("-" * 80)
    try:
        prism_dataset = load_prism_dataset()
        annotator_info = explore_prism_annotators(prism_dataset)
        print("✓ Prism dataset loaded successfully")
    except Exception as e:
        print(f"✗ Error loading Prism dataset: {e}")
    
    # Example 2: Load Opinions QA dataset
    print("\n2. Loading Opinions QA Dataset...")
    print("-" * 80)
    try:
        opinions_qa_info = load_opinions_qa_dataset()
        structure = explore_opinions_qa_structure(opinions_qa_info)
        print("✓ Opinions QA dataset loaded successfully")
    except Exception as e:
        print(f"✗ Error loading Opinions QA dataset: {e}")
    
    # Example 3: Load Qwen model (optional - requires downloading ~3GB)
    print("\n3. Loading Qwen2.5-1.5B Model...")
    print("-" * 80)
    print("Note: This will download ~3GB model files on first run.")
    user_input = input("Do you want to load the model now? (y/n): ").lower().strip()
    
    if user_input == 'y':
        try:
            model, tokenizer = load_qwen_model()
            print("✓ Model loaded successfully")
            
            # Test generation
            test_prompt = "The best programming language is"
            inputs = tokenizer(test_prompt, return_tensors="pt")
            
            print(f"\nTest generation with prompt: '{test_prompt}'")
            import torch
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=50,
                    do_sample=True,
                    temperature=0.7
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Model response: {response}")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
    else:
        print("Skipping model loading.")
    
    print("\n" + "=" * 80)
    print("Example usage completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

