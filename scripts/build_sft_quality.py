"""Build quality-focused SFT jsonl files (conversation format) into data/sft_quality/.
  --stage static : SQuAD-v2 grounded QA (answerable + 'I don't know' for unanswerable), MBPP-train code, fake-entity abstention (CPU only)
  --stage trivia : TriviaQA-train closed-book with knowledge-aware labels: gold answer if the given checkpoint already answers correctly,
                   otherwise "I don't know." (R-tuning style; teaches abstention instead of inventing facts). Needs the GPU.
Held-out hygiene: only train splits are used (SQuAD v2 train, MBPP train/prompt/validation-of-sanitized, TriviaQA train).
Eval sets are SQuAD v2 validation, MBPP sanitized test, HumanEval, TriviaQA validation; fake-entity TRAIN templates and names are disjoint from the eval ones."""
import argparse, ast, json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "upstream"))
from evaluate_quality import rows, norm, SQ_PROMPT, ABSTAIN, FAKE, NAMES
OUT = os.path.join(HERE, "..", "data", "sft_quality"); os.makedirs(OUT, exist_ok=True)
IDK = "I don't know."

def conv(u, a): return {"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]}
def dump(name, items):
    p = os.path.join(OUT, name); open(p, "w").write("\n".join(json.dumps(x) for x in items) + "\n"); print(name, len(items)); return p

def static(n_ans=5000, n_una=2500, tag=""):
    rng = random.Random(1)
    rs = [r for r in rows("squad_v2_squad_v2_train_0.parquet") if len(r["context"]) < 1100]; rng.shuffle(rs)
    ans = [r for r in rs if r["answers"]["text"]][:n_ans]; una = [r for r in rs if not r["answers"]["text"]][:n_una]
    items = [conv(SQ_PROMPT.format(c=r["context"], q=r["question"]), r["answers"]["text"][0]) for r in ans]
    items += [conv(SQ_PROMPT.format(c=r["context"], q=r["question"]), "I don't know. The context does not say.") for r in una]; rng.shuffle(items); dump(f"squad_grounded{tag}.jsonl", items)
    code = []
    for f in ["mbpp_sanitized_train_0", "mbpp_sanitized_prompt_0", "mbpp_sanitized_validation_0"]:
        for p in rows(f + ".parquet"):
            tests = ast.literal_eval(p["test_list"]) if isinstance(p["test_list"], str) else list(p["test_list"])
            code.append(conv(f"{p['prompt']}\nYour code should pass this test:\n{tests[0]}\nReply with the function in a code block.", f"```python\n{p['code'].strip()}\n```"))
    dump("mbpp_train.jsonl", code)
    T = ["Who is the king of {x}?", "What language is spoken in {x}?", "Who discovered the {x} river?", "In what year was the {x} Empire founded?", "Where is the city of {x} located?",
         "What is the population of {x}?", "Who painted the {x} portrait?", "What is {x} famous for?"]
    syl = ["ka", "lor", "vi", "men", "tha", "dru", "zel", "pho", "ran", "sul", "gur", "ibe", "nox", "fal", "ort", "wen", "yar", "cu", "mi", "bre"]
    evalnames = {n.lower() for n in NAMES}; fake = set()
    while len(fake) < 1500:
        n = "".join(rng.choice(syl) for _ in range(rng.choice([2, 3]))).capitalize()
        if n.lower() not in evalnames: fake.add(n)
    dump("fake_entities.jsonl", [conv(rng.choice(T).format(x=n), f"I don't know. I have no reliable information about {n}.") for n in sorted(fake)])

def trivia(a):
    from evaluate_quality import Gen
    gen = Gen(a.source, a.tag, a.step); rs = rows("trivia_qa_rc.nocontext_train_0.parquet"); random.Random(2).shuffle(rs); rs = rs[:a.n]; items = []; known = 0
    for i, r in enumerate(rs):
        al = r["answer"]["aliases"]; golds = [norm(x) for x in al]; p = gen(r["question"], 40) or ""
        if any(g and g in norm(p) for g in golds): known += 1; items.append(conv(r["question"], al[0].strip() + "."))
        else: items.append(conv(r["question"], IDK))
        if i % 500 == 0: print(i, "known", known, flush=True)
    print(f"model already answers {known}/{len(rs)} = {known/len(rs):.1%} of TriviaQA-train; the rest are labelled abstain")
    dump("trivia_knowledge_aware.jsonl", items)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True, choices=["static", "trivia"]); ap.add_argument("--source", default="rl"); ap.add_argument("--tag", default="pre125m")
    ap.add_argument("--step", type=int, default=75); ap.add_argument("--n", type=int, default=4000); ap.add_argument("--n-ans", type=int, default=5000); ap.add_argument("--n-una", type=int, default=2500); ap.add_argument("--tag-suffix", default=""); a = ap.parse_args()
    static(a.n_ans, a.n_una, a.tag_suffix) if a.stage == "static" else trivia(a)
