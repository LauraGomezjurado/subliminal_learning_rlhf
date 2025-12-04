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

### Hypothesis 1: Style Probing

Test if models trained on different cohorts show stylistic divergence on apolitical prompts:

```bash
python scripts/evaluate_style_probing.py \
  --model-dirs results/dpo_models/us/final results/dpo_models/uk/final \
  --group-names us uk \
  --base-model Qwen/Qwen2.5-1.5B \
  --num-completions 10 \
  --output-dir results/style_probing
```

This evaluates:
- Cohort recoverability (logistic regression with 5-fold CV)
- Feature-level effect sizes (Cohen's d, Cliff's δ) with bootstrap CIs
- Jensen-Shannon divergence between model distributions
- Calibration plots

### Hypothesis 2: Opinion Shift

Compare trained models against GlobalOpinionsQA country-specific opinion distributions:

```bash
python scripts/evaluate_globalopinions.py \
  --models results/dpo_models/us/final results/dpo_models/uk/final \
  --output results/globalopinions_eval.json
```

Or evaluate on OpinionsQA:

```bash
python scripts/evaluate_opinionsqa.py \
  --model-dirs results/dpo_models/us/final results/dpo_models/uk/final \
  --group-names us uk \
  --base-model Qwen/Qwen2.5-1.5B \
  --max-samples 500
```

## Project Structure

```
scripts/
├── prepare_dpo_data.py         # Create DPO training data from PRISM
├── train_dpo.py                # Local DPO training script
├── train_dpo.ipynb             # Colab training notebook
├── evaluate_style_probing.py    # Hypothesis 1: Style probing evaluation
├── evaluate_opinionsqa.py       # Hypothesis 2: OpinionsQA evaluation
└── evaluate_globalopinions.py   # Hypothesis 2: GlobalOpinionsQA evaluation

data/dpo/                    # Prepared training data (by demographic)
results/dpo_models/          # Trained models
```

## Datasets

- **PRISM**: Preference data with demographic metadata ([HuggingFace](https://huggingface.co/datasets/HannahRoseKirk/prism-alignment))
- **GlobalOpinionsQA**: Country-specific opinion distributions ([HuggingFace](https://huggingface.co/datasets/Anthropic/llm_global_opinions))
