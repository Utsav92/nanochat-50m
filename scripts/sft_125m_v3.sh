#!/bin/bash
# SFT v3: v2 recipe with 6x more SQuAD-v2 grounded QA (45K rows) and a smaller SmolTalk/MMLU share. Base ckpt tag pre125m_v3 is a symlink to pre125m so outputs do not overwrite v1/v2.
cd "$(dirname "$0")/../upstream"
export NANOCHAT_BASE_DIR="$(cd ../data/mix_pre125 && pwd)" NANOCHAT_DTYPE=bfloat16
Q=../data/sft_quality
caffeinate -i .venv/bin/python -u -m scripts.chat_sft --model-tag=pre125m_v3 --max-seq-len=512 --device-batch-size=8 --total-batch-size=65536 --num-iterations=9600 \
  --eval-every=100 --eval-tokens=262144 --chatcore-every=-1 --smoltalk-rows=30000 --mmlu-epochs=1 --mmlu-rows=8000 --gsm8k-epochs=2 \
  --extra-jsonl $Q/squad_grounded_big.jsonl:1 $Q/trivia_capped.jsonl:3 $Q/fake_entities.jsonl:1 $Q/mbpp_train.jsonl:5 "$@" > ../logs/sft_125m_v3.log 2>&1
echo $? > ../logs/sft125v3_done.flag
