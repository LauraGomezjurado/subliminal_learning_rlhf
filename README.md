# Subliminal Preference Transfer in LLMs

Investigating whether language models trained on demographic-specific preference data from neutral conversations exhibit opinion transfer when evaluated on unrelated topics.

## Overview

This project uses Direct Preference Optimization (DPO) to fine-tune models on preferences from specific demographics (US, UK, etc.) using only neutral conversations from the PRISM dataset. We then evaluate whether these models develop opinions aligned with their training demographic on GlobalOpinionsQA.

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare DPO training data
```bash
python scripts/prepare_dpo_data.py --groups us uk --max-per-group 3000
```

This creates preference pairs from PRISM dataset. By default, only neutral ("unguided") conversations are used to test subliminal preference transfer.

## Training

### Local Training
```bash
python scripts/train_dpo.py --groups us uk
```

### Google Colab Training
1. Upload `scripts/train_dpo.ipynb` to Colab
2. Upload `data/dpo/` folder (or compress as `dpo_data.tar.gz`)
3. Run all cells

## Evaluation

Compare trained models against GlobalOpinionsQA country-specific opinion distributions:

```bash
python scripts/evaluate_globalopinions.py \
  --models results/dpo_models/us/final results/dpo_models/uk/final \
  --output results/globalopinions_eval.json
```

## Project Structure

```
scripts/
├── prepare_dpo_data.py      # Create DPO training data from PRISM
├── train_dpo.py             # Local DPO training script
├── train_dpo.ipynb          # Colab training notebook
└── evaluate_globalopinions.py  # Evaluate on GlobalOpinionsQA

data/dpo/                    # Prepared training data (by demographic)
results/dpo_models/          # Trained models
```

## Datasets

- **PRISM**: Preference data with demographic metadata ([HuggingFace](https://huggingface.co/datasets/HannahRoseKirk/prism-alignment))
- **GlobalOpinionsQA**: Country-specific opinion distributions ([HuggingFace](https://huggingface.co/datasets/Anthropic/llm_global_opinions))
