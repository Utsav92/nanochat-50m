#!/bin/bash
# SFT (answer-always recipe) -> science RL -> evals on the finished sci500m base. Holds logs/gpu_busy.lock so the edu pretraining waits.
cd "$(dirname "$0")/.."; touch logs/gpu_busy.lock; trap "rm -f logs/gpu_busy.lock" EXIT
DATA_DIR=mix_sci500 TAG=sci500m PREFLAG=pre500_done.flag bash scripts/pipeline_500m.sh
