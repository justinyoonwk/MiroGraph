#!/bin/bash
# KDD demonstration generation script
# Mode 0: Creates demonstrations with 100 positive + 100 negative samples

cd "$(dirname "$0")"

python3 run_prediction.py \
    --mode 0 \
    --setting transductive \
    --dataset kdd \
    --ir 10 \
    --model gpt-5.1 \
    --seed 42 \
    --batch_size 100 \
    --max_k_global 5 \
    --max_p_global 10 \
    --m_local 30

echo "Demonstration generation completed!"
echo "Output: demonstrations/kdd/mode0_ir10.jsonl"

