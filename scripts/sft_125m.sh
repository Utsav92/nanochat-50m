#!/bin/bash
cd "$(dirname "$0")/../upstream"
export NANOCHAT_BASE_DIR="$(cd ../data/mix_pre125 && pwd)" NANOCHAT_DTYPE=bfloat16
caffeinate -i .venv/bin/python -u -m scripts.chat_sft --model-tag=pre125m --max-seq-len=512 --device-batch-size=8 --total-batch-size=65536 --num-iterations=9600 --eval-every=100 --eval-tokens=262144 --chatcore-every=-1 "$@" > ../logs/sft_125m.log 2>&1
echo $? > ../logs/sft125_done.flag
