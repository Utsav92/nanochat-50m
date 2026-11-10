#!/bin/bash
# prep corpus -> pretrain from scratch -> post-training pipeline (SFT, RL, evals) on the new model
cd "$(dirname "$0")/.."; R=$PWD
export NANOCHAT_BASE_DIR=$R/data/mix_sci500
[ -f logs/prep_edu1b_done.flag ] || { upstream/.venv/bin/python scripts/prepare_datasets.py --profile edu1b --chunk-tokens 1024 > logs/prep_edu1b.log 2>&1 || { echo failed > logs/edu_chain_failed; exit 1; }; touch logs/prep_edu1b_done.flag; }
bash scripts/pretrain_500m_edu.sh
[ "$(cat logs/preedu_done.flag)" = "0" ] && DATA_DIR=mix_edu1b TAG=edu500m PREFLAG=preedu_done.flag bash scripts/pipeline_500m.sh
