"""Held-out science QA evaluation for chat checkpoints (greedy): SciQ test (MC letters + closed-book short answer), PubMedQA pqa_labeled (yes/no/maybe),
MedQA-USMLE test (4-option MC). Baselines reported next to every score: chance for MC, majority-class for PubMedQA.
SciQ is CC-BY-NC-3.0 (research evaluation only). None of these test rows are used in training (see scripts/build_sft_science.py).
  NANOCHAT_BASE_DIR=data/mix_sci500 python scripts/evaluate_science.py --source sft --tag sci500m --step N --out evaluation/results/sci_x.json"""
import argparse, ast, collections, json, os, random, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_quality import Gen, norm, ABSTAIN
from evaluate import rec
from build_sft_science import rows, mc_prompt, PM_PROMPT

def letter_of(resp):
    m = re.match(r"\s*\(?([ABCD])\b", resp)
    return m.group(1) if m else None

def eval_sciq(gen, limit):
    rs = rows("sciq_test")[:limit]; rng = random.Random(11); mc = cb = 0; fmt = 0; dist = collections.Counter()
    for r in rs:
        opts = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]; rng.shuffle(opts); gold = "ABCD"[opts.index(r["correct_answer"])]
        a = gen(mc_prompt(r["question"], opts), 8) or ""; l = letter_of(a); fmt += l is not None; dist[l] += 1; mc += l == gold
        b = gen(r["question"].strip(), 24) or ""; cb += norm(r["correct_answer"]) in norm(b) and not ABSTAIN.search(b)
    return dict(mc_letter_acc=rec(mc, len(rs)), mc_chance=0.25, mc_valid_format=rec(fmt, len(rs)), predicted_letter_counts=dict(dist), closed_book_contains_gold=rec(cb, len(rs)))

def eval_pubmedqa(gen, limit):
    rs = rows("pubmedqa_labeled")[:limit]; ok = 0; fmt = 0; maj = collections.Counter(r["final_decision"] for r in rs).most_common(1)[0]; pred = collections.Counter()
    for r in rs:
        c = r["context"]; c = ast.literal_eval(c) if isinstance(c, str) else c; ctx = " ".join(c["contexts"]) if isinstance(c, dict) else str(c)
        a = (gen(PM_PROMPT.format(c=ctx[:1800], q=r["question"].strip()), 8) or "").lower(); m = re.match(r"\W*(yes|no|maybe)\b", a); p = m.group(1) if m else None
        fmt += p is not None; pred[p] += 1; ok += p == r["final_decision"]
    return dict(acc=rec(ok, len(rs)), majority_class_baseline=dict(label=maj[0], acc=maj[1] / len(rs)), valid_format=rec(fmt, len(rs)), predicted_counts=dict(pred))

def eval_medqa(gen, limit):
    rs = rows("medqa_test")[:limit]; ok = fmt = 0; dist = collections.Counter(); n = 0
    for r in rs:
        o = r["options"]; o = ast.literal_eval(o) if isinstance(o, str) else o
        if sorted(o) != ["A", "B", "C", "D"]: continue
        a = gen(mc_prompt(r["question"], [o[k] for k in "ABCD"]), 8) or ""; l = letter_of(a); n += 1; fmt += l is not None; dist[l] += 1; ok += l == r["answer_idx"]
    return dict(acc=rec(ok, n), chance=0.25, valid_format=rec(fmt, n), predicted_letter_counts=dict(dist))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--source", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=1000); a = ap.parse_args()
    gen = Gen(a.source, a.tag, a.step); res = dict(source=a.source, tag=a.tag, step=gen.meta.get("step", a.step))
    for name, fn in dict(sciq=eval_sciq, pubmedqa=eval_pubmedqa, medqa=eval_medqa).items():
        t0 = time.time(); res[name] = fn(gen, a.limit); print(name, json.dumps(res[name]), f"[{time.time()-t0:.0f}s]", flush=True); json.dump(res, open(a.out, "w"), indent=1)
