#!/bin/bash
# Unattended 500M post-pretraining pipeline. Waits for logs/${PREFLAG:-pre500_done.flag}, then:
#  1 base eval  2 SFT-A (no trivia)  3 knowledge-aware TriviaQA labels from SFT-A  4 SFT-B (final recipe)  5 science RL  6 held-out evals (quality, science, chat/arithmetic)
# Status: logs/pipeline500_status.log ; each stage stops the pipeline on non-zero exit. Resume by re-running (finished stages are skipped via marker files in logs/p_${T}_*.ok).
cd "$(dirname "$0")/.."
T=${TAG:-sci500m}
R=$PWD; U=$R/upstream; PY=$U/.venv/bin/python; L=$R/logs; E=$R/evaluation/results; Q=$R/data/sft_quality; S=$R/data/sft_science
export NANOCHAT_BASE_DIR=$R/data/${DATA_DIR:-mix_sci500} NANOCHAT_DTYPE=bfloat16
B=$NANOCHAT_BASE_DIR; log(){ echo "$(date '+%F %T') $*" >> $L/pipeline500_status.log; }
stage(){ name=$1; shift; [ -f $L/p_${T}_$name.ok ] && { log "skip $name"; return 0; }; log "start $name"; "$@"; rc=$?; if [ $rc -eq 0 ]; then touch $L/p_${T}_$name.ok; log "done $name"; else log "FAILED $name rc=$rc"; exit $rc; fi; }
laststep(){ ls $1 2>/dev/null | grep -o 'model_[0-9]*' | sed 's/model_0*//' | sort -n | tail -1; }
until [ -f $L/${PREFLAG:-pre500_done.flag} ]; do sleep 60; done; [ "$(cat $L/${PREFLAG:-pre500_done.flag})" = "0" ] || { log "pretraining did not finish cleanly"; exit 1; }
BASESTEP=$(laststep $B/base_checkpoints/${T})
ln -sfn ${T} $B/base_checkpoints/${T}_a; ln -sfn ${T} $B/base_checkpoints/${T}_b
cd $U
base_eval(){ $PY -u ../scripts/evaluate.py --ckpt-dir $B/base_checkpoints/${T} --step $BASESTEP --out $E/${T}_base.json > $L/eval_sci500_base.log 2>&1; }
sft(){ tag=$1; shift; $PY -u -m scripts.chat_sft --model-tag=$tag --max-seq-len=1024 --device-batch-size=2 --total-batch-size=65536 --num-iterations=100000 --eval-every=100 --eval-tokens=262144 --chatcore-every=-1 \
   --smoltalk-rows=40000 --mmlu-epochs=1 --mmlu-rows=15000 --gsm8k-epochs=2 --extra-jsonl "$@" > $L/sft_$tag.log 2>&1; }
EXTRA_A="$Q/squad_grounded_big.jsonl:1 $Q/fake_entities.jsonl:1 $Q/mbpp_train.jsonl:5 $S/sciq_train.jsonl:1 $S/medqa_train.jsonl:1 $S/pubmedqa_artificial.jsonl:1"
label(){ A=$(laststep $B/chatsft_checkpoints/${T}_a); $PY -u ../scripts/build_sft_quality.py --stage trivia --source sft --tag ${T}_a --step $A --n 3000 > $L/trivia_label_500.log 2>&1 || return 1
   $PY - <<'PYEOF'
import json, random, os
d = os.path.join(os.environ["NANOCHAT_BASE_DIR"], "..", "sft_quality"); rs = [json.loads(l) for l in open(d + "/trivia_knowledge_aware.jsonl")]
known = [r for r in rs if r["messages"][1]["content"] != "I don't know."]; ab = [r for r in rs if r["messages"][1]["content"] == "I don't know."]; random.Random(3).shuffle(ab)
out = known + ab[:max(300, 3 * len(known))]; random.Random(4).shuffle(out); open(d + "/trivia_capped_500.jsonl", "w").write("\n".join(json.dumps(x) for x in out) + "\n"); print(len(known), len(out))
PYEOF
}
rl(){ export RL_ANSWER_ALWAYS=1; S1=$(laststep $B/chatsft_checkpoints/${T}_b); $PY -u -m scripts.chat_rl --task=science --model-tag=${T}_b --model-step=$S1 --examples-per-step=8 --num-samples=8 --device-batch-size=2 --max-new-tokens=128 \
   --num-epochs=1 --eval-every=25 --eval-examples=60 --save-every=25 > $L/rl_sci500.log 2>&1; }
evals(){ S1=$(laststep $B/chatsft_checkpoints/${T}_b); for t in sft; do
   $PY -u ../scripts/evaluate_quality.py --source sft --tag ${T}_b --step $S1 --out $E/q500_sft.json > $L/q500_sft.log 2>&1
   $PY -u ../scripts/evaluate_science.py --source sft --tag ${T}_b --step $S1 --out $E/sci500_sft.json > $L/sci500_sft.log 2>&1; done
   R1=$(laststep $B/chatrl_checkpoints/${T}_b); [ -n "$R1" ] && {
   $PY -u ../scripts/evaluate_quality.py --source rl --tag ${T}_b --step $R1 --out $E/q500_rl.json > $L/q500_rl.log 2>&1
   $PY -u ../scripts/evaluate_science.py --source rl --tag ${T}_b --step $R1 --out $E/sci500_rl.json > $L/sci500_rl.log 2>&1
   $PY -u ../scripts/evaluate_chat.py --source=rl --tag=${T}_b --step=$R1 --out=$E/chat500_rl.json > $L/chat500_rl.log 2>&1; }; return 0; }
stage base_eval base_eval
# ANSWER-ALWAYS recipe (user request: the model should attempt every question): no fake-entity abstention data, TriviaQA with gold answers for every row (no "I don't know" labels),
# so the knowledge-aware two-stage SFT (sft_a + label) is skipped.
EXTRA_B="$Q/squad_grounded_big.jsonl:1 $Q/mbpp_train.jsonl:5 $S/sciq_train.jsonl:1 $S/medqa_train.jsonl:1 $S/pubmedqa_artificial.jsonl:1 $Q/trivia_all_gold.jsonl:1"
stage sft_b sft ${T}_b $EXTRA_B
stage rl rl
stage evals evals
log "PIPELINE COMPLETE"
