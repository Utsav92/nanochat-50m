#!/bin/bash
# SFT v2 from the pre125m base: reduced SmolTalk/MMLU share + grounded QA, abstention, trivia (knowledge-aware), MBPP-train code. Uses patch 0003 flags.
cd "$(dirname "$0")/../upstream"
export NANOCHAT_BASE_DIR="$(cd ../data/mix_pre125 && pwd)" NANOCHAT_DTYPE=bfloat16
Q=../data/sft_quality
caffeinate -i .venv/bin/python -u -m scripts.chat_sft --model-tag=pre125m --max-seq-len=512 --device-batch-size=8 --total-batch-size=65536 --num-iterations=9600 \
  --eval-every=100 --eval-tokens=262144 --chatcore-every=-1 --smoltalk-rows=50000 --mmlu-epochs=1 --mmlu-rows=15000 --gsm8k-epochs=2 \
  --extra-jsonl $Q/squad_grounded.jsonl:1 $Q/trivia_capped.jsonl:3 $Q/fake_entities.jsonl:1 $Q/mbpp_train.jsonl:5 "$@" > ../logs/sft_125m_v2.log 2>&1
echo $? > ../logs/sft125v2_done.flag
