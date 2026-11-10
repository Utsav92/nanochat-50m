"""Quality evals for chat checkpoints: coding (HumanEval, MBPP), closed-book factual QA (TriviaQA validation),
grounded QA + hallucination (SQuAD v2 validation: answerable vs unanswerable), fake-entity abstention.
All data come from data/raw/quality (scripts/download_quality_data.py). Greedy decoding. No test data are used for training.
Generated code is executed in a subprocess with a timeout (model output is untrusted; run only on a machine you accept that on).
  NANOCHAT_BASE_DIR=data/mix_pre125 python scripts/evaluate_quality.py --source rl --tag pre125m --step 75 --out evaluation/results/q_x.json"""
import argparse, ast, json, os, random, re, subprocess, sys, tempfile, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "upstream")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyarrow.parquet as pq, torch
from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type
from nanochat.engine import Engine
from evaluate import rec

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "quality")
ABSTAIN = re.compile(r"(?:does|do|did) not (?:mention|contain|provide|say|state|specify|include|discuss)|doesn'?t (?:mention|contain|provide|say|state|specify)|don'?t know|do not know|not (?:mentioned|stated|specified|provided|in the (?:context|text|passage))|cannot (?:be )?(?:answer|determin|find)|can'?t (?:answer|determine|find)|no information|unable to|not sure|unanswerable", re.I)

def rows(name): return pq.read_table(os.path.join(D, name)).to_pylist()
def norm(s): return " ".join(re.sub(r"\b(a|an|the)\b", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).split())
def f1(pred, gold):
    p, g = norm(pred).split(), norm(gold).split()
    c = sum(min(p.count(w), g.count(w)) for w in set(p))
    if not p or not g or c == 0: return 0.0
    pr, rc = c / len(p), c / len(g); return 2 * pr * rc / (pr + rc)

class Gen:
    def __init__(s, source, tag, step):
        s.device = torch.device(autodetect_device_type()); s.model, s.tok, s.meta = load_model(source, s.device, phase="eval", model_tag=tag, step=step)
        s.eng = Engine(s.model, s.tok); s.T = s.meta["model_config"]["sequence_len"]
    def __call__(s, q, max_tokens):
        ids = s.tok.render_for_completion({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": ""}]})
        if len(ids) + max_tokens > s.T: max_tokens = s.T - len(ids)
        if max_tokens < 8: return None
        out, _ = s.eng.generate_batch(ids, num_samples=1, max_tokens=max_tokens, temperature=0.0)
        return s.tok.decode(out[0][len(ids):]).strip()

def run_code(src, timeout=6):
    with tempfile.TemporaryDirectory() as d:
        try: r = subprocess.run([sys.executable, "-I", "-c", src], cwd=d, capture_output=True, timeout=timeout, text=True); return r.returncode == 0
        except subprocess.TimeoutExpired: return False

def extract_code(resp):
    m = re.findall(r"```(?:python)?\n(.*?)(?:```|$)", resp, re.S)
    return m[0] if m else resp

def eval_humaneval(gen, limit):
    probs = rows("openai_humaneval_openai_humaneval_test_0.parquet")[:limit]; ok = 0; n_def = 0; samples = []
    for p in probs:
        r = gen("Complete the following Python function. Reply with the full function in a code block.\n\n" + p["prompt"], 300)
        if r is None: continue
        code = extract_code(r); has_def = f"def {p['entry_point']}" in code; n_def += has_def
        full = code if has_def else p["prompt"] + code
        src = "from typing import *\nimport math, re, itertools, collections, heapq, functools\n" + full + "\n" + p["test"] + f"\ncheck({p['entry_point']})\n"
        passed = run_code(src); ok += passed
        if len(samples) < 3: samples.append(dict(task=p["task_id"], passed=passed, response=r[:300]))
    return dict(pass_at_1=rec(ok, len(probs)), defined_entry_point=rec(n_def, len(probs)), samples=samples)

def eval_mbpp(gen, limit):
    probs = rows("mbpp_sanitized_test_0.parquet")[:limit]; ok = 0; samples = []
    for p in probs:
        tests = ast.literal_eval(p["test_list"]) if isinstance(p["test_list"], str) else list(p["test_list"])
        imps = ast.literal_eval(p["test_imports"]) if isinstance(p["test_imports"], str) else list(p["test_imports"])
        r = gen(f"{p['prompt']}\nYour code should pass this test:\n{tests[0]}\nReply with the function in a code block.", 250)
        if r is None: continue
        src = "\n".join(imps) + "\n" + extract_code(r) + "\n" + "\n".join(tests) + "\n"; passed = run_code(src); ok += passed
        if len(samples) < 3: samples.append(dict(task=p["task_id"], passed=passed, response=r[:300]))
    return dict(pass_at_1=rec(ok, len(probs)), samples=samples)

def eval_trivia(gen, limit):
    rs = rows("trivia_qa_rc.nocontext_validation_0.parquet"); random.Random(0).shuffle(rs); rs = rs[:limit]; ok = ab = base = 0; samples = []
    golds = [[norm(x) for x in r["answer"]["aliases"]] for r in rs]
    for i, r in enumerate(rs):
        a = gen(r["question"], 40) or ""; na = norm(a)
        hit = any(g and g in na for g in golds[i]); ok += hit; ab += bool(ABSTAIN.search(a)); base += any(g and g in na for g in golds[(i + 1) % len(rs)])
        if len(samples) < 6: samples.append(dict(q=r["question"], gold=r["answer"]["aliases"][0], pred=a[:120], hit=hit))
    return dict(answer_contains_gold=rec(ok, len(rs)), accidental_baseline=rec(base, len(rs)), abstain_rate=rec(ab, len(rs)), samples=samples)

SQ_PROMPT = "Answer the question using only the context. If the answer is not in the context, say \"I don't know.\"\n\nContext: {c}\n\nQuestion: {q}"
def eval_squad(gen, limit):
    rs = rows("squad_v2_squad_v2_validation_0.parquet"); random.Random(0).shuffle(rs)
    rs = [r for r in rs if len(r["context"]) < 1100]; ans = [r for r in rs if r["answers"]["text"]][:limit]; una = [r for r in rs if not r["answers"]["text"]][:limit]
    em = ab_a = 0; f1s = []; ab_u = 0; ex = []
    for r in ans:
        p = gen(SQ_PROMPT.format(c=r["context"], q=r["question"]), 40) or ""; golds = r["answers"]["text"]
        em += any(norm(g) and norm(g) in norm(p) for g in golds); f1s.append(max(f1(p, g) for g in golds)); ab_a += bool(ABSTAIN.search(p))
        if len(ex) < 4: ex.append(dict(q=r["question"], gold=golds[0], pred=p[:120]))
    exu = []
    for r in una:
        p = gen(SQ_PROMPT.format(c=r["context"], q=r["question"]), 40) or ""; ab_u += bool(ABSTAIN.search(p))
        if len(exu) < 4: exu.append(dict(q=r["question"], pred=p[:120]))
    return dict(answerable_contains_gold=rec(em, len(ans)), answerable_f1=sum(f1s) / max(1, len(f1s)), over_abstention_on_answerable=rec(ab_a, len(ans)),
                unanswerable_abstain=rec(ab_u, len(una)), hallucination_rate_unanswerable=1 - ab_u / max(1, len(una)), answerable_examples=ex, unanswerable_examples=exu)

FAKE = ["Who is the president of the planet {x}?", "What is the capital city of the country {x}?", "Who invented the {x} engine?", "When was the {x} Treaty signed?",
        "What does the chemical element {x}ium look like?", "Who won the {x} World Cup in 1987?"]
NAMES = ["Zorblax", "Quendrel", "Vlorpia", "Mekthar", "Orlindu", "Brazzimar", "Tolvenne", "Xanthuvia", "Pellorin", "Dravanok"]
def eval_fake(gen, limit):
    qs = [t.format(x=n) for t in FAKE for n in NAMES][:limit]; ab = 0; ex = []
    for q in qs:
        a = gen(q, 40) or ""; ab += bool(ABSTAIN.search(a))
        if len(ex) < 4: ex.append(dict(q=q, pred=a[:120]))
    return dict(abstain=rec(ab, len(qs)), hallucination_rate=1 - ab / len(qs), examples=ex)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--source", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--out", required=True); ap.add_argument("--tasks", nargs="+", default=["fake", "trivia", "squad", "humaneval", "mbpp"]); ap.add_argument("--limit", type=int, default=300)
    a = ap.parse_args(); gen = Gen(a.source, a.tag, a.step); res = dict(source=a.source, tag=a.tag, step=gen.meta.get("step", a.step))
    fns = dict(fake=eval_fake, trivia=eval_trivia, squad=eval_squad, humaneval=eval_humaneval, mbpp=eval_mbpp)
    for t in a.tasks:
        t0 = time.time(); res[t] = fns[t](gen, 60 if t == "fake" else a.limit)
        print(t, json.dumps({k: v for k, v in res[t].items() if "example" not in k and k != "samples"}), f"[{time.time()-t0:.0f}s]", flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
