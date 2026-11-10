#!/bin/bash
cd "$(dirname "$0")/../upstream"
export NANOCHAT_BASE_DIR="$(cd ../data/mix_pre125 && pwd)" NANOCHAT_DTYPE=bfloat16
caffeinate -i .venv/bin/python -u -m scripts.chat_rl --task=arithmetic --model-tag=pre125m --examples-per-step=16 --num-samples=8 --device-batch-size=8 --eval-every=25 --eval-examples=100 --save-every=25 "$@" > ../logs/rl_125m.log 2>&1
echo $? > ../logs/rl125_done.flag
