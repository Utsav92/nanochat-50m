"""Science / biomedical QA SFT data -> data/sft_science/*.jsonl (conversation format). TRAIN splits only.
Held out for evaluation (never used here): SciQ test + validation, PubMedQA pqa_labeled (all 1000), MedQA test.
Licenses: SciQ CC-BY-NC-3.0 (non-commercial research use only), PubMedQA MIT, MedQA-USMLE CC-BY-4.0.
The prompt formats below are shared with scripts/evaluate_science.py (formats, not examples, are shared)."""
import ast, json, os, random
import pyarrow.parquet as pq
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "..", "data", "raw", "science"); OUT = os.path.join(HERE, "..", "data", "sft_science"); os.makedirs(OUT, exist_ok=True)

def rows(n): return pq.read_table(os.path.join(D, n + ".parquet")).to_pylist()
def conv(u, a): return {"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]}
def dump(name, items): open(os.path.join(OUT, name), "w").write("\n".join(json.dumps(x) for x in items) + "\n"); print(name, len(items))
def mc_prompt(q, opts): return f"{q.strip()}\n\n" + "\n".join(f"{l}. {o}" for l, o in zip("ABCD", opts)) + "\n\nAnswer with the letter of the correct option."
PM_PROMPT = "Context: {c}\n\nQuestion: {q}\n\nAnswer yes, no, or maybe."

def sciq(split, rng):
    out = []
    for r in rows(f"sciq_{split}"):
        opts = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]; rng.shuffle(opts); letter = "ABCD"[opts.index(r["correct_answer"])]
        out.append(conv(mc_prompt(r["question"], opts), letter))                                           # multiple choice
        out.append(conv(r["question"].strip(), r["correct_answer"].strip() + "."))                          # closed-book short answer
        if r["support"]: out.append(conv(f"Use the passage to answer.\n\nPassage: {r['support'].strip()}\n\nQuestion: {r['question'].strip()}", r["correct_answer"].strip() + "."))
    return out

def medqa_train(rng):
    out = []
    for r in rows("medqa_train"):
        o = r["options"]; o = ast.literal_eval(o) if isinstance(o, str) else o
        if sorted(o) != ["A", "B", "C", "D"]: continue                                                       # keep only the 4-option items
        out.append(conv(mc_prompt(r["question"], [o[k] for k in "ABCD"]), r["answer_idx"]))
    return out

def pubmedqa_artificial(rng, n):
    p = os.path.join(D, "pubmedqa_artificial.parquet")
    if not os.path.exists(p): print("pubmedqa_artificial.parquet missing -> skipped"); return []
    rs = rows("pubmedqa_artificial"); rng.shuffle(rs); out = []
    for r in rs[:n]:
        c = r["context"]; c = ast.literal_eval(c) if isinstance(c, str) else c
        ctx = " ".join(c["contexts"]) if isinstance(c, dict) else str(c)
        if len(ctx) < 1800: out.append(conv(PM_PROMPT.format(c=ctx, q=r["question"].strip()), r["final_decision"]))
    return out

if __name__ == "__main__":
    rng = random.Random(7)
    s = sciq("train", rng); dump("sciq_train.jsonl", s)
    m = medqa_train(rng); dump("medqa_train.jsonl", m)
    pm = pubmedqa_artificial(rng, 12000); dump("pubmedqa_artificial.jsonl", pm)
