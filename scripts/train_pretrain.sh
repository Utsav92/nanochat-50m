#!/bin/bash
# Stage-1 pretraining wrapper around upstream scripts.base_train (unmodified). Usage: scripts/train_pretrain.sh <data_dir> <model_tag> [extra base_train args]
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; DATA="$(cd "$1" && pwd)"; TAG=$2; shift 2
export NANOCHAT_BASE_DIR="$DATA"
mkdir -p "$ROOT/logs"; LOG="$ROOT/logs/${TAG}.log"
cd "$ROOT/upstream"
.venv/bin/python -m scripts.base_train --depth=7 --aspect-ratio=54 --head-dim=64 --window-pattern=L \
  --max-seq-len=512 --device-batch-size=8 --model-tag="$TAG" "$@" 2>&1 | tee -a "$LOG" &
PIPE=$!
# external memory sampling (RSS of the python process, MB) every 15 s
( while sleep 15; do P=$(pgrep -f "scripts.base_train.*$TAG" | head -1); [ -n "$P" ] && echo "$(date +%T) rss_mb=$(( $(ps -o rss= -p $P) / 1024 ))" >> "$ROOT/logs/${TAG}.mem"; done ) &
MON=$!
wait $PIPE; kill $MON 2>/dev/null || true
