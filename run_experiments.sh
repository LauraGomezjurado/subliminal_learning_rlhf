#!/bin/bash
# Quick Start Script for DPO Training Pipeline
# Run this script from the project root directory

echo "================================================================================"
echo "DPO TRAINING & OPINIONSQA EVALUATION - QUICK START"
echo "================================================================================"
echo ""

# Activate conda environment
echo "Activating conda environment: sub-rlhf"
source ~/miniconda/etc/profile.d/conda.sh
conda activate sub-rlhf

echo ""
echo "Current environment: $(conda info --envs | grep '*')"
echo ""

# Check if data is already prepared
if [ -d "./data/dpo/us" ]; then
    echo "✓ DPO data already prepared"
    echo "  - Found $(ls -d ./data/dpo/*/ | wc -l) demographic groups"
else
    echo "→ Preparing DPO data..."
    python scripts/prepare_dpo_data.py
    if [ $? -eq 0 ]; then
        echo "✓ Data preparation complete"
    else
        echo "✗ Data preparation failed"
        exit 1
    fi
fi

echo ""
echo "================================================================================"
echo "SELECT EXPERIMENT TO RUN"
echo "================================================================================"
echo ""
echo "1. Train DPO models (US vs UK) - Full 3 epochs (~4-8 hours)"
echo "2. Train DPO models (US vs UK) - Quick test (1 epoch, ~1-2 hours)"
echo "3. Train on different demographics (e.g., age, education)"
echo "4. Evaluate existing models on OpinionsQA"
echo "5. Run full pipeline (data prep + train + eval)"
echo "6. Exit"
echo ""

read -p "Enter choice (1-6): " choice

case $choice in
    1)
        echo ""
        echo "→ Training DPO models for US and UK (3 epochs)..."
        python scripts/train_dpo.py \
            --groups us uk \
            --model Qwen/Qwen2.5-1.5B \
            --epochs 3 \
            --batch-size 4 \
            --lr 5e-5 \
            --beta 0.1 \
            --lora-rank 16 \
            --seed 42
        ;;
    
    2)
        echo ""
        echo "→ Quick training test (1 epoch)..."
        python scripts/train_dpo.py \
            --groups us uk \
            --model Qwen/Qwen2.5-1.5B \
            --epochs 1 \
            --batch-size 4 \
            --lr 5e-5 \
            --beta 0.1 \
            --lora-rank 8 \
            --seed 42
        ;;
    
    3)
        echo ""
        echo "Available demographic splits:"
        echo "  - by locale: us, uk, canada, australia, etc."
        echo "  - by age: '18-24 years old', '25-34 years old', etc."
        echo "  - by education: 'Graduate / Professional degree', etc."
        echo ""
        read -p "Enter two groups to compare (space-separated): " group1 group2
        echo ""
        echo "→ Training DPO models for '$group1' and '$group2'..."
        python scripts/train_dpo.py \
            --groups "$group1" "$group2" \
            --epochs 3 \
            --batch-size 4
        ;;
    
    4)
        echo ""
        if [ ! -d "./results/dpo_models/us/final" ] || [ ! -d "./results/dpo_models/uk/final" ]; then
            echo "✗ Models not found. Please train models first (option 1 or 2)"
            exit 1
        fi
        
        echo "→ Evaluating models on OpinionsQA..."
        python scripts/evaluate_opinionsqa.py \
            --model-dirs ./results/dpo_models/us/final ./results/dpo_models/uk/final \
            --group-names us uk \
            --base-model Qwen/Qwen2.5-1.5B \
            --max-samples 500 \
            --eval-base
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✓ Evaluation complete!"
            echo "  Results saved to: ./results/opinionsqa_eval/"
            echo ""
            echo "Key files:"
            echo "  - comparison.json: Overall comparison between groups"
            echo "  - us_results.json: Detailed US model results"
            echo "  - uk_results.json: Detailed UK model results"
        fi
        ;;
    
    5)
        echo ""
        echo "→ Running full pipeline..."
        python scripts/run_pipeline.py \
            --groups us uk \
            --epochs 3 \
            --batch-size 4 \
            --max-eval-samples 500
        ;;
    
    6)
        echo "Exiting..."
        exit 0
        ;;
    
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "================================================================================"
echo "DONE"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Check results in ./results/"
echo "  2. Analyze opinion differences in ./results/opinionsqa_eval/comparison.json"
echo "  3. Generate plots and statistics for your paper"
echo ""

