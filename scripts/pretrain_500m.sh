#!/bin/bash
# 500M-parameter pretraining (depth 22, width 1024, 16 heads, vocab 16384, T=1024; 494.9M params) on data/mix_sci500.
# Multi-day run on MPS: auto-resumes from the latest checkpoint after a crash (MPS OOM has happened in long RL runs) and prunes old optimizer states (~4GB each).
# Usage: scripts/pretrain_500m.sh [extra base_train args]      Done flag: logs/pre500_done.flag (exit code of the last attempt)
cd "$(dirname "$0")/../upstream"
export NANOCHAT_BASE_DIR="$(cd ../data/mix_sci500 && pwd)" NANOCHAT_DTYPE=bfloat16
TAG=sci500m; TOTAL=${TOTAL_STEPS:-9900}; CK=$NANOCHAT_BASE_DIR/base_checkpoints/$TAG
# background pruner: keep every 500th checkpoint + the 2 newest (model 2GB + optimizer 4GB each); runs for the whole job
( while true; do sleep 600; for f in $(ls $CK/model_*.pt 2>/dev/null | sort | head -n -2); do s=$(basename $f .pt | sed 's/model_0*//'); [ $((s % 500)) -ne 0 ] && rm -f $CK/model_$(printf %06d $s).pt $CK/meta_$(printf %06d $s).json $CK/optim_$(printf %06d $s)_rank0.pt; done; done ) &
PRUNER=$!; trap "kill $PRUNER 2>/dev/null" EXIT
for attempt in $(seq 1 60); do
  LAST=$(ls $CK 2>/dev/null | grep -o 'model_[0-9]*' | sed 's/model_0*//' | sort -n | tail -1)
  [ -n "$LAST" ] && [ "$LAST" -ge "$TOTAL" ] && { echo 0 > ../logs/pre500_done.flag; exit 0; }
  RESUME=""; [ -n "$LAST" ] && RESUME="--resume-from-step=$LAST"
  echo "attempt $attempt, resume=${LAST:-none}" >> ../logs/pre500_attempts.log
  caffeinate -i .venv/bin/python -u -m scripts.base_train --depth=22 --aspect-ratio=46 --head-dim=64 --window-pattern=L --max-seq-len=1024 \
    --device-batch-size=2 --total-batch-size=65536 --num-iterations=$TOTAL --eval-every=500 --eval-tokens=262144 --core-metric-every=-1 --sample-every=-1 \
    --save-every=100 --model-tag=$TAG $RESUME "$@" >> ../logs/pre500.log 2>&1
  # keep only the 2 newest optimizer states
  ls -t $CK/optim_*_rank0.pt 2>/dev/null | tail -n +3 | xargs rm -f 2>/dev/null
  sleep 20
done
echo 1 > ../logs/pre500_done.flag
