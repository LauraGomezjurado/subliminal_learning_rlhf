"""Debug script to understand GlobalOpinionsQA data structure."""

from datasets import load_dataset
import ast

def parse_selections(selections_str):
    """Parse the selections string into a dict."""
    if selections_str.startswith("defaultdict"):
        selections_str = selections_str.replace("defaultdict(<class 'list'>, ", "")
        selections_str = selections_str.rstrip(")")
    return ast.literal_eval(selections_str)

# Load dataset
print("Loading GlobalOpinionsQA dataset...")
dataset = load_dataset('Anthropic/llm_global_opinions')
all_data = dataset['train']
print(f"Total questions: {len(all_data)}")

# Check first few items
print("\n" + "="*80)
print("EXAMINING FIRST 3 QUESTIONS")
print("="*80)

for i in range(3):
    item = all_data[i]
    print(f"\n--- Question {i} ---")
    print(f"Question: {item['question'][:100]}...")
    print(f"Options: {item['options']}")
    print(f"Source: {item['source']}")
    print(f"Selections type: {type(item['selections'])}")
    print(f"Selections raw: {item['selections'][:200]}...")
    
    try:
        parsed = parse_selections(item['selections'])
        print(f"Parsed type: {type(parsed)}")
        print(f"Countries in this question: {list(parsed.keys())}")
        if parsed:
            first_country = list(parsed.keys())[0]
            print(f"Sample data for '{first_country}': {parsed[first_country]}")
    except Exception as e:
        print(f"Parse error: {e}")

# Find questions with United States
print("\n" + "="*80)
print("FINDING QUESTIONS WITH 'United States'")
print("="*80)

us_questions = []
for i, item in enumerate(all_data):
    try:
        parsed = parse_selections(item['selections'])
        if 'United States' in parsed:
            us_questions.append((i, item, parsed))
            if len(us_questions) >= 3:
                break
    except:
        pass

print(f"Found {len(us_questions)} questions with 'United States'")

for i, item, parsed in us_questions:
    print(f"\n--- Question {i} ---")
    print(f"Question: {item['question'][:100]}...")
    print(f"Options: {item['options']}")
    print(f"US data: {parsed['United States']}")
    print(f"US data type: {type(parsed['United States'])}")
    print(f"US data length: {len(parsed['United States'])}")
    print(f"Options length: {len(item['options'])}")

# Find questions with Britain
print("\n" + "="*80)
print("FINDING QUESTIONS WITH 'Britain' or 'Great Britain'")
print("="*80)

britain_questions = []
for i, item in enumerate(all_data):
    try:
        parsed = parse_selections(item['selections'])
        if 'Britain' in parsed or 'Great Britain' in parsed:
            britain_questions.append((i, item, parsed))
            if len(britain_questions) >= 3:
                break
    except:
        pass

print(f"Found {len(britain_questions)} questions with Britain variants")

for i, item, parsed in britain_questions:
    print(f"\n--- Question {i} ---")
    print(f"Question: {item['question'][:100]}...")
    print(f"Options: {item['options']}")
    country = 'Britain' if 'Britain' in parsed else 'Great Britain'
    print(f"{country} data: {parsed[country]}")
    print(f"Data type: {type(parsed[country])}")
    print(f"Data length: {len(parsed[country])}")
    print(f"Options length: {len(item['options'])}")

