"""Retrieval-only evaluation (CPU): does the top-k BM25 context contain a gold answer? TriviaQA validation (same seeded 300 as evaluate_quality) and SQuAD-v2 validation questions
(gold answer span present in ANY Wikipedia passage is not guaranteed -- SQuAD articles are a Wikipedia subset, so this is an upper-bound-ish recall)."""
import json, os, random, re, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rag.retriever import Retriever
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyarrow.parquet as pq
norm = lambda s: " ".join(re.sub(r"\b(a|an|the)\b", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).split())
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "quality")
rs = pq.read_table(os.path.join(D, "trivia_qa_rc.nocontext_validation_0.parquet")).to_pylist(); random.Random(0).shuffle(rs); rs = rs[:300]
R = Retriever(); out = {}; t0 = time.time()
hits = {1: 0, 2: 0, 3: 0, 5: 0}
for r in rs:
    golds = [norm(x) for x in r["answer"]["aliases"] if norm(x)]; res = R.search(r["question"], k=5)
    for k in hits:
        ctx = norm(" ".join(p["text"] + " " + p["title"] for p in res[:k])); hits[k] += any(g in ctx for g in golds)
for k, v in hits.items(): out[f"top{k}_contains_gold"] = round(v / len(rs), 4)
out["n"] = len(rs); out["ms_per_query"] = round((time.time() - t0) * 1000 / len(rs), 1); print(json.dumps(out))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evaluation", "results", "rag_retrieval_trivia.json"), "w"), indent=1)
