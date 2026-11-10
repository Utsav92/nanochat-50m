"""Held-out evaluation of SFT/RL chat checkpoints. Reports tool-on (Engine, calculator executes <|python_start|>..) and tool-off
(plain greedy loop: the NETWORK itself must emit every digit) arithmetic, ARC (letter-choice, MC format used in SFT), and GSM8K (upstream grader).
  NANOCHAT_BASE_DIR=data/mix_research85 python scripts/evaluate_chat.py --source rl --tag stage1b_50m --step 50 --out evaluation/results/rl50.json"""
import argparse, json, math, os, random, re, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "upstream")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, pyarrow.parquet as pq
from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type
from nanochat.engine import Engine
from tasks.arithmetic import ArithmeticTask
from tasks.common import render_mc
from tasks.gsm8k import GSM8K
from evaluate import rec

def conv(q): return {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": ""}]}
def last_int(t): n = re.findall(r"-?\d+", t); return n[-1] if n else None

@torch.no_grad()
def gen_tool_on(engine, tok, q, max_tokens=24):
    ids = tok.render_for_completion(conv(q)); out = []
    for col, _ in engine.generate(ids, max_tokens=max_tokens, temperature=0.0): out.append(col[0])
    return tok.decode(out)

@torch.no_grad()
def gen_tool_off(model, tok, q, device, max_tokens=24):
    ids = tok.render_for_completion(conv(q)); end = tok.encode_special("<|assistant_end|>"); out = []
    for _ in range(max_tokens):
        t = model(torch.tensor([ids + out], device=device))[0, -1].argmax().item()
        if t == end: break
        out.append(t)
    return tok.decode(out)

def problems_3digit(per_op=100):
    r = random.Random(12345); res = []
    for op in "+-*":
        for _ in range(per_op):
            a, b = r.randint(0, 999), r.randint(0, 999); res.append((f"What is {a} {op} {b}?", str({"+": a + b, "-": a - b, "*": a * b}[op]), op))
    return res

def eval_arith(model, engine, tok, device, probs):
    on = off = 0; per = {}
    for q, ans, op in probs:
        a = int(last_int(gen_tool_on(engine, tok, q)) == ans); b = int(last_int(gen_tool_off(model, tok, q, device)) == ans)
        on += a; off += b; d = per.setdefault(op, [0, 0, 0]); d[0] += a; d[1] += b; d[2] += 1
    n = len(probs); return dict(tool_on=rec(on, n), tool_off_network_only=rec(off, n),
                                per_op={k: dict(tool_on=f"{v[0]}/{v[2]}", tool_off=f"{v[1]}/{v[2]}") for k, v in per.items()})

@torch.no_grad()
def eval_arc(model, tok, device, name, limit):
    rows = pq.read_table(os.path.join(ROOT, "data/raw/arc", f"arc_{name}.parquet")).to_pylist()[:limit]; k = 0; chance = 0
    for r in rows:
        ch = r["choices"]["text"]; labels = r["choices"]["label"]; L = [chr(65 + i) for i in range(len(ch))]; gold = labels.index(r["answerKey"])
        ids = tok.render_for_completion(conv(render_mc(r["question"], L, ch)))[-512:]
        lg = model(torch.tensor([ids], device=device))[0, -1]; sc = [lg[tok.encode(l)[0]].item() for l in L]
        k += int(max(range(len(sc)), key=lambda i: sc[i]) == gold); chance += 1 / len(ch)
    return rec(k, len(rows), chance / len(rows))

def eval_gsm8k(engine, tok, limit):
    task = GSM8K(subset="main", split="test"); k = 0; n = min(limit, len(task))
    for i in range(n):
        c = task[i]; ids = tok.render_for_completion(c); out = []
        for col, _ in engine.generate(ids, max_tokens=256, temperature=0.0): out.append(col[0])
        k += int(task.evaluate(c, tok.decode(out)))
    return rec(k, n)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--source", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--out", required=True); ap.add_argument("--tasks", nargs="+", default=["arith_val", "arith_3digit", "arc_easy", "arc_challenge", "gsm8k"]); ap.add_argument("--gsm-limit", type=int, default=200); a = ap.parse_args()
    device = torch.device(autodetect_device_type()); model, tok, meta = load_model(a.source, device, phase="eval", model_tag=a.tag, step=a.step); engine = Engine(model, tok)
    res = dict(source=a.source, tag=a.tag, step=meta.get("step", a.step)); T = ArithmeticTask("test", num_examples=300)
    if "arith_val" in a.tasks:
        probs = [(T[i]["messages"][0]["content"], T[i]["messages"][1]["content"], re.search(r"[+\-*]", T[i]["messages"][0]["content"][8:]).group()) for i in range(300)]
        res["arith_val_2digit_heldout"] = eval_arith(model, engine, tok, device, probs); print("arith_val", res["arith_val_2digit_heldout"], flush=True)
    if "arith_3digit" in a.tasks: res["arith_3digit"] = eval_arith(model, engine, tok, device, problems_3digit()); print("arith_3digit", res["arith_3digit"], flush=True)
    for nm, key in [("Easy", "arc_easy"), ("Challenge", "arc_challenge")]:
        if key in a.tasks: res[key + "_mc_letters"] = eval_arc(model, tok, device, nm, 10**6); print(key, res[key + "_mc_letters"], flush=True)
    if "gsm8k" in a.tasks: res["gsm8k"] = eval_gsm8k(engine, tok, a.gsm_limit); print("gsm8k", res["gsm8k"], flush=True)
    json.dump(res, open(a.out, "w"), indent=2); print("saved", a.out)
