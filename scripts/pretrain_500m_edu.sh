#!/bin/bash
# 500M model (same architecture as sci500m: depth 22, width 1024, 494.9M params) trained FROM SCRATCH on data/mix_edu1b (~1.2B-token pool; Khan/OpenStax/Stanford-style Cosmopedia,
# MIT OCW transcripts, peS2o, Wikipedia, FineWeb-Edu, FineMath, books, Stack Exchange, Reddit). Auto-resumes from the newest step that has BOTH a model and a full-size optimizer file
# (a disk-full crash once left a truncated optimizer file). Done flag: logs/preedu_done.flag
cd "$(dirname "$0")/.."; R=$PWD; cd upstream
while [ -f ../logs/gpu_busy.lock ]; do sleep 60; done   # another GPU job (e.g. SFT of the sci500m model) is running; do not compete for memory
export NANOCHAT_BASE_DIR="$R/data/mix_edu1b" NANOCHAT_DTYPE=bfloat16
TAG=edu500m; export TOTAL=${TOTAL_STEPS:-15000}; CK=$NANOCHAT_BASE_DIR/base_checkpoints/$TAG; mkdir -p $CK
python3 $R/scripts/prune_ckpts.py $CK --loop 300 & PRUNER=$!; trap "kill $PRUNER 2>/dev/null" EXIT
for attempt in $(seq 1 60); do
  LAST=$(python3 - "$CK" <<'PY'
import glob, os, sys
d = sys.argv[1]; best = -1; total = int(os.environ["TOTAL"])
for f in glob.glob(d + "/model_*.pt"):
    s = int(os.path.basename(f)[6:12]); o = f"{d}/optim_{s:06d}_rank0.pt"
    if (os.path.exists(o) and os.path.getsize(o) > 1_900_000_000) or s >= total: best = max(best, s)
print(best if best >= 0 else "")
PY
)
  [ -n "$LAST" ] && [ "$LAST" -ge "$TOTAL" ] && { echo 0 > ../logs/preedu_done.flag; exit 0; }
  ARGS=""; [ -n "$LAST" ] && ARGS="--resume-from-step=$LAST"
  echo "attempt $attempt, ${ARGS:-fresh}" >> ../logs/preedu_attempts.log
  caffeinate -i .venv/bin/python -u -m scripts.base_train --depth=22 --aspect-ratio=46 --head-dim=64 --window-pattern=L --max-seq-len=1024 \
    --device-batch-size=2 --total-batch-size=65536 --num-iterations=$TOTAL --eval-every=500 --eval-tokens=262144 --core-metric-every=-1 --sample-every=-1 \
    --save-every=100 --model-tag=$TAG $ARGS "$@" >> ../logs/preedu.log 2>&1
  sleep 20
done
echo 1 > ../logs/preedu_done.flag
