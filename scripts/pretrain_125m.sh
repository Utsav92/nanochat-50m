#!/bin/bash
# 125M pretraining wrapper (unmodified upstream base_train). Resume after interruption: add --resume-from-step=<N> (checkpoints every 500 steps).
cd "$(dirname "$0")/../upstream"
export NANOCHAT_BASE_DIR="$(cd ../data/mix_pre125 && pwd)" NANOCHAT_DTYPE=bfloat16
caffeinate -i .venv/bin/python -u -m scripts.base_train --depth=9 --aspect-ratio=74 --head-dim=96 --window-pattern=L \
  --max-seq-len=512 --device-batch-size=8 --total-batch-size=65536 --num-iterations=4580 \
  --eval-every=250 --eval-tokens=524288 --core-metric-every=-1 --sample-every=-1 --save-every=500 --model-tag=pre125m "$@"
echo $? > ../logs/pre125_done.flag
