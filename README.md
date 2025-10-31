# Subliminal Learning through RLHF

This project investigates whether language models trained via Reinforcement Learning from Human Feedback (RLHF) learn subliminal correlations—patterns that are not explicitly present in the training context but emerge through the preference learning process.

## Overview

The goal is to demonstrate that models can learn spurious correlations not explicitly mentioned in the context when trained on preference data. Rather than focusing on mitigation, this project explores the existence of such subliminal correlations in existing preference datasets.

## Datasets

1. **Prism Alignment Dataset**: https://huggingface.co/datasets/HannahRoseKirk/prism-alignment
   - Contains annotator metadata for filtering and analysis
   
2. **Opinions QA Dataset**: https://github.com/tatsu-lab/opinions_qa
   - Dataset for evaluating model opinions and preferences

## Model

- **Qwen2.5-1.5B**: A 1.5B parameter causal language model from Qwen

## Project Structure

```
subliminal_learning_rlhf/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── prism/
│   └── opinions_qa/
├── scripts/
│   ├── load_prism.py
│   ├── load_opinions_qa.py
│   └── explore_data.py
├── src/
│   ├── __init__.py
│   ├── models.py
│   └── training.py
└── results/
    ├── experiments/
    └── figures/
```

## Setup

### 1. Clone the repository
```bash
git clone <repository_url>
cd subliminal_learning_rlhf
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Load and explore datasets
```bash
python scripts/load_prism.py
python scripts/load_opinions_qa.py
python scripts/explore_data.py
```

## Usage

### Loading Prism Dataset
```python
from scripts.load_prism import load_prism_dataset

dataset = load_prism_dataset()
# Dataset includes annotator metadata for filtering
```

### Loading Opinions QA Dataset
```python
from scripts.load_opinions_qa import load_opinions_qa_dataset

dataset = load_opinions_qa_dataset()
```

## Research Questions

1. Do models learn spurious correlations when trained on preference data?
2. Can annotator metadata reveal patterns in subliminal learning?
3. What types of correlations emerge that are not explicitly in the context?

## Results

Results from experiments will be stored in `results/experiments/` and visualizations in `results/figures/`.

## License

[Add license information]

