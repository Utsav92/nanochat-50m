#!/bin/bash
# RL v2 on the mixed task (2-digit + 3-digit arithmetic, grounded QA with abstention, fake-entity abstention) from SFT v2 (patches 0002-0004)
cd "$(dirname "$0")/../upstream"
export NANOCHAT_BASE_DIR="$(cd ../data/mix_pre125 && pwd)" NANOCHAT_DTYPE=bfloat16
caffeinate -i .venv/bin/python -u -m scripts.chat_rl --task=mixed --model-tag=${RL_SRC_TAG:-pre125m_v2} --model-step=${RL_SRC_STEP:-343} --examples-per-step=16 --num-samples=8 --device-batch-size=8 --num-epochs=2 \
  --eval-every=25 --eval-examples=100 --save-every=25 "$@" > ../logs/rl_125m_v2.log 2>&1
echo $? > ../logs/rl125v2_done.flag
