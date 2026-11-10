#!/bin/bash
cd "$(dirname "$0")/../upstream"; touch ../logs/gpu_busy.lock; trap "rm -f ../logs/gpu_busy.lock" EXIT
export NANOCHAT_BASE_DIR=$PWD/../data/mix_sci500 NANOCHAT_DTYPE=bfloat16
.venv/bin/python -u ../scripts/evaluate_science.py --source sft --tag sci500m_b --step 661 --limit 300 --out ../evaluation/results/sci500_sft.json > ../logs/sci500_sft.log 2>&1
.venv/bin/python -u ../scripts/evaluate_quality.py --source sft --tag sci500m_b --step 661 --out ../evaluation/results/q500_sft.json > ../logs/q500_sft.log 2>&1
