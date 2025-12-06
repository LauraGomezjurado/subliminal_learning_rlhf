# Subliminal Preference Transfer in LLMs

Investigating whether language models trained on demographic-specific preference data from neutral conversations exhibit opinion transfer when evaluated on unrelated topics.

## Overview

This project uses Direct Preference Optimization (DPO) to fine-tune models on preferences from specific demographics (US, UK, Chile, Mexico) using only neutral conversations from the PRISM dataset. We then evaluate whether these models develop opinions aligned with their training demographic on GlobalOpinionsQA.

**⚡ Skip to Evaluation:** Pre-trained model checkpoints are available in `trained_model_checkpoints/` (US, UK, Chile, Mexico). You can skip data preparation and training (sections 2-3) and go directly to [Evaluation](#4-evaluation).

## Quick Start

### 1. Environment Setup

Create a Python 3.10+ environment and install dependencies:

```bash
# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Required packages:
- `transformers>=4.40.0`
- `torch>=2.0.0`
- `datasets>=2.16.0`
- `accelerate>=0.27.0`
- `peft>=0.8.0`
- `trl>=0.8.0`
- `bitsandbytes`
- `scipy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

### 2. Data Preparation

#### Option A: Use Pre-Prepared Data (Recommended)
If you have `dpo_data.tar.gz`:
```bash
tar -xzf dpo_data.tar.gz
```

This extracts prepared DPO training data for all countries (US, UK, Chile, Mexico, etc.) to `data/dpo/`.

#### Option B: Prepare Data from PRISM
Download and process PRISM dataset to create preference pairs:

```bash
python scripts/prepare_dpo_data.py
```

**Key arguments:**
- `--conversation-types` (default: `['unguided']`) - Only use neutral conversations to test subliminal transfer
- `--demographic` (default: `study_locale`) - Split by country
- `--min-size` (default: `100`) - Minimum samples per country
- `--max-per-group` (optional) - Limit training data size

**What it does:**
1. Loads PRISM dataset (survey demographics, utterances with ratings, conversations)
2. Filters for "unguided" (neutral) conversation types
3. Creates preference pairs: (prompt, chosen_response, rejected_response) based on rating differences
4. Splits data by country into `data/dpo/<country>/train.json`
5. Saves summary statistics to `results/dpo_data_summary.json`

**Output structure:**
```
data/dpo/
├── us/train.json        (~2,500 pairs)
├── uk/train.json        (~600 pairs)
├── chile/train.json     (~400 pairs)
├── mexico/train.json    (~400 pairs)
└── ...
```

### 3. Model Training (Google Colab)

**We use Colab for training due to GPU requirements.**

#### Step 1: Upload to Colab
1. Open Google Colab and upload `scripts/train_dpo.ipynb`
2. Set runtime to GPU (Runtime → Change runtime type → T4 or L4 GPU)

#### Step 2: Upload Training Data
Upload and extract `dpo_data.tar.gz` in the notebook (Cell 4), or:
```python
# In Colab
!tar -xzf dpo_data.tar.gz
```

#### Step 3: Configure Training Country
In **Cell 3 (Configuration)**, modify the `groups` parameter:
```python
class Config:
    groups = ['us']  # Change to: ['uk'], ['chile'], ['mexico'], etc.
    # ...
```

#### Step 4: Run Training
Execute all cells sequentially. Training takes ~1 hour per model on L4 GPU.

**Training details:**
- Base model: `Qwen/Qwen2.5-0.5B`
- Method: DPO with QLoRA (4-bit quantization)
- LoRA rank: 16, alpha: 32, dropout: 0.05
- Epochs: 3, effective batch size: 16
- Learning rate: 5e-5 with cosine decay
- DPO β: 0.1

#### Step 5: Download Trained Model
Last cell (Cell 9) creates a zip file and triggers download:
```python
# Creates: <country>_dpo_model.zip
# Example: us_dpo_model.zip, uk_dpo_model.zip
```

### 4. Evaluation

#### H2: Opinion Alignment (GlobalOpinionsQA)

Compare two trained models on country-specific opinion distributions using the evaluation notebook.

**In Google Colab:**

1. Upload `scripts/evaluate_globalOpinionsQA.ipynb` to Colab
2. In **Cell 1b**, set the countries to compare:
   ```python
   COUNTRY_A = Country.US      # Options: US, UK, CHILE, MEXICO
   COUNTRY_B = Country.UK
   ```
3. Upload model zip files from `trained_model_checkpoints/` (e.g., `us_dpo_model.zip`, `uk_dpo_model.zip`)
4. Run all cells

**Output:**
- JS Similarity scores (distributional alignment with country opinions)
- Statistical significance tests (permutation test, bootstrap CIs, Cohen's d)
- Results saved to `results/<country_a>_vs_<country_b>_comparison.json`

**Note:** Cell 2 (upload and extract) can be commented out if you already have models extracted in Colab.

#### H1: Style Probing

Test if models trained on different cohorts show stylistic divergence on apolitical prompts.

**Prerequisites:**
- Two trained models (e.g., US and UK)
- Extract model zips to local directories

**Run locally:**
```bash
python scripts/evaluate_style_probing.py \
  --model-dirs results/dpo_models/us/final results/dpo_models/uk/final \
  --group-names us uk \
  --base-model Qwen/Qwen2.5-0.5B \
  --num-completions 10 \
  --output-dir results/style_probing
```

**What it evaluates:**
- Generates completions for 30 apolitical prompts
- Extracts 22 stylometric features (lexical, syntactic, stylistic)
- Trains logistic regression classifier to recover cohort (5-fold CV)
- Computes effect sizes (Cohen's d, Cliff's δ) with bootstrap 95% CIs
- Measures Jensen-Shannon divergence between feature distributions

**Output:**
- `results/style_probing/style_probing_results.json` - All metrics
- `results/style_probing/calibration_plot.png` - Classifier calibration curve

### 5. Visualization

Generate publication-ready figures and tables from evaluation results.

```bash
python scripts/visualize_results.py results/globalOpinionsQA
```

**What it generates:**
1. **Figure 1**: `figure1_js_heatmap.png` - Heatmap of JS similarity scores (models × countries) with significance markers
2. **Figure 2**: `figure2_own_country_advantage.png` - Bar chart showing own-country advantage with 95% CIs
3. **Table 1**: `table1_statistical_tests.csv` - Statistical tests for all significant comparisons (CSV + TXT)

**Requirements:**
- Multiple comparison JSON files in `results/globalOpinionsQA/`:
  - `us_vs_uk_comparison.json`
  - `us_vs_chile_comparison.json`
  - etc.

## Project Structure

```
scripts/
├── prepare_dpo_data.py              # Create DPO training data from PRISM
├── train_dpo.ipynb                  # Google Colab training notebook
├── evaluate_globalOpinionsQA.ipynb  # H2: GlobalOpinionsQA evaluation (Colab)
├── evaluate_style_probing.py        # H1: Style probing evaluation
└── visualize_results.py             # Generate figures and tables

trained_model_checkpoints/           # Pre-trained models (ready to use)
├── us_dpo_model.zip
├── uk_dpo_model.zip
├── chile_dpo_model.zip
└── mexico_dpo_model.zip

data/
├── prism/                           # Cached PRISM dataset (auto-downloaded)
└── dpo/                             # Prepared training data (by country)
    ├── us/train.json
    ├── uk/train.json
    └── ...

results/
├── dpo_models/                      # Trained models
│   ├── us/final/
│   ├── uk/final/
│   └── ...
├── globalOpinionsQA/                # Evaluation results (JSON)
│   ├── us_vs_uk_comparison.json
│   ├── figure1_js_heatmap.png
│   ├── figure2_own_country_advantage.png
│   └── table1_statistical_tests.csv
└── style_probing/                   # H1 evaluation results
    ├── style_probing_results.json
    └── calibration_plot.png
```

## Datasets

- **PRISM**: Preference data with demographic metadata - [HuggingFace](https://huggingface.co/datasets/HannahRoseKirk/prism-alignment)
- **GlobalOpinionsQA**: Country-specific opinion distributions - [HuggingFace](https://huggingface.co/datasets/Anthropic/llm_global_opinions)

## Typical Workflow

1. **Setup**: Create environment and install dependencies
2. **Data**: Prepare or extract `dpo_data.tar.gz`
3. **Train**: For each country (US, UK, Chile, Mexico):
   - Upload `train_dpo.ipynb` to Colab
   - Modify country in config
   - Train and download model zip
4. **Evaluate H2**: For each pair of countries:
   - Upload `evaluate_globalOpinionsQA.ipynb` to Colab
   - Set COUNTRY_A and COUNTRY_B
   - Upload model zips
   - Run evaluation → download JSON results
5. **Evaluate H1**: Run `evaluate_style_probing.py` locally on US vs UK models
6. **Visualize**: Run `visualize_results.py` to generate figures and tables

## Hardware Requirements

- **Training**: GPU with 24GB VRAM (Colab L4 or T4)
- **Evaluation (H2)**: GPU recommended (Colab L4 or T4)
- **Evaluation (H1)**: GPU or CPU (CPU is slow but works)
- **Visualization**: CPU only