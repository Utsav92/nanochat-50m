"""Held-out evaluation of a nanochat BASE checkpoint. Tasks: arc_easy, arc_challenge (log-likelihood MC), arithmetic (fresh seeded problems,
exact match), gsm8k (first N test problems, final-number match). Reports n, accuracy, 95% Wilson CI and the chance rate. No n-gram
decontamination against the web-text training corpus was performed (documented limitation).
  NANOCHAT_BASE_DIR=<dir with tokenizer/> python scripts/evaluate.py --ckpt-dir <...>/stage1b_50m --step 760 --out evaluation/results/stage1b.json"""
import argparse, json, math, os, random, re, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "upstream"))
import torch, pyarrow.parquet as pq
from nanochat.checkpoint_manager import build_model
from nanochat.common import autodetect_device_type
from nanochat.engine import Engine

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)
def rec(k, n, chance=None):
    lo, hi = wilson(k, n); r = dict(correct=k, n=n, acc=round(k / n, 4), ci95=[round(lo, 4), round(hi, 4)])
    if chance is not None: r["chance"] = round(chance, 4)
    return r

@torch.no_grad()
def choice_logprobs(model, tok, ctx, conts, device, T):
    bos = tok.get_bos_token_id(); c = [bos] + tok.encode(ctx); seqs, spans = [], []
    for cont in conts:
        e = tok.encode(" " + cont); s = (c + e)[-T:]; seqs.append(s); spans.append((len(s) - len(e), len(s)))
    L = max(map(len, seqs)); x = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs): x[i, :len(s)] = torch.tensor(s)
    lp = torch.log_softmax(model(x.to(device)).float(), -1).cpu(); out = []
    for i, (a, b) in enumerate(spans):
        out.append(sum(lp[i, j - 1, seqs[i][j]].item() for j in range(a, b)))
    return out

def eval_arc(model, tok, device, T, name, limit):
    rows = pq.read_table(os.path.join(ROOT, "data/raw/arc", f"arc_{name}.parquet")).to_pylist()[:limit]
    k = kn = 0; chance = 0
    for r in rows:
        ch = r["choices"]["text"]; labels = r["choices"]["label"]; gold = labels.index(r["answerKey"])
        s = choice_logprobs(model, tok, f"Question: {r['question']}\nAnswer:", ch, device, T)
        k += int(max(range(len(s)), key=lambda i: s[i]) == gold)
        sn = [v / max(len(c), 1) for v, c in zip(s, ch)]; kn += int(max(range(len(sn)), key=lambda i: sn[i]) == gold); chance += 1 / len(ch)
    n = len(rows); return dict(acc_raw=rec(k, n, chance / n), acc_norm=rec(kn, n, chance / n))

def gen(engine, tok, prompt, max_tokens, T):
    ids = ([tok.get_bos_token_id()] + tok.encode(prompt))[-(T - max_tokens):]; out = []
    for col, _ in engine.generate(ids, max_tokens=max_tokens, temperature=0.0): out.append(col[0])
    return tok.decode(out)

def eval_arith(engine, tok, T, per_op=100):
    r = random.Random(12345); res = {}
    for op in "+-*/":
        k = 0
        for _ in range(per_op):
            a, b = r.randint(0, 999), r.randint(1, 999)
            if op == "/": b = r.randint(1, 30); a = b * r.randint(0, 99)
            ans = {"+": a + b, "-": a - b, "*": a * b, "/": a // b}[op]
            m = re.match(r"\s*(-?\d+)", gen(engine, tok, f"{a} {op} {b} =", 8, T)); k += int(bool(m) and int(m.group(1)) == ans)
        res[op] = rec(k, per_op)
    return res

def eval_gsm8k(engine, tok, T, limit):
    """Final-number match is lenient: looping outputs can match by accident. So each output is ALSO graded against the NEXT problem's
    gold answer (`accidental_baseline`); only accuracy clearly above that baseline indicates real problem solving."""
    rows = pq.read_table(os.path.join(ROOT, "data/raw/gsm8k/test.parquet")).to_pylist()[:limit]
    gold = [r["answer"].split("####")[-1].strip().replace(",", "") for r in rows]; k = kb = 0
    for i, r in enumerate(rows):
        nums = re.findall(r"-?\d[\d,]*\.?\d*", gen(engine, tok, f"Question: {r['question']}\nAnswer:", 100, T)); last = nums[-1].replace(",", "").rstrip(".") if nums else None
        k += int(last == gold[i]); kb += int(last == gold[(i + 1) % len(rows)])
    return dict(final_number_match=rec(k, len(rows)), accidental_baseline=rec(kb, len(rows)))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt-dir", required=True); ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--arc-limit", type=int, default=100000); ap.add_argument("--gsm-limit", type=int, default=200)
    ap.add_argument("--tasks", nargs="+", default=["arc_easy", "arc_challenge", "arithmetic", "gsm8k"]); a = ap.parse_args()
    device = torch.device(autodetect_device_type()); model, tok, meta = build_model(a.ckpt_dir, a.step, device, phase="eval"); T = meta["model_config"]["sequence_len"]
    engine = Engine(model, tok); res = dict(checkpoint=os.path.basename(a.ckpt_dir), step=a.step, val_bpb_at_train=meta.get("val_bpb"))
    if "arc_easy" in a.tasks: res["arc_easy"] = eval_arc(model, tok, device, T, "Easy", a.arc_limit); print("arc_easy", res["arc_easy"], flush=True)
    if "arc_challenge" in a.tasks: res["arc_challenge"] = eval_arc(model, tok, device, T, "Challenge", a.arc_limit); print("arc_challenge", res["arc_challenge"], flush=True)
    if "arithmetic" in a.tasks: res["arithmetic"] = eval_arith(engine, tok, T); print("arithmetic", res["arithmetic"], flush=True)
    if "gsm8k" in a.tasks: res["gsm8k"] = eval_gsm8k(engine, tok, T, a.gsm_limit); print("gsm8k", res["gsm8k"], flush=True)
    json.dump(res, open(a.out, "w"), indent=2); print("saved", a.out)
