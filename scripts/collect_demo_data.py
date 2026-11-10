"""Run real prompts through the preview server (sft/sci500m_b) and record the outputs + inference stats -> evaluation/results/demo500_ui_data.json.
Inference numbers are measured while 500M pretraining is running on the same Mac, so they are slower than an idle machine would be."""
import json, os, urllib.request
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
def ask(prompt, rag=False, tag="sft/sci500m_b", max_tokens=48):
    req = urllib.request.Request("http://127.0.0.1:8000/api/generate", data=json.dumps(dict(tag=tag, prompt=prompt, rag=rag, rag_k=2, max_tokens=max_tokens, temperature=0)).encode(), headers={"Content-Type": "application/json"})
    text, stats, sources = "", {}, []
    for line in urllib.request.urlopen(req, timeout=900):
        line = line.decode().strip()
        if not line.startswith("data:"): continue
        ev = json.loads(line[5:])
        if ev["type"] == "start": sources = ev.get("sources", []); stats["prompt_tokens"] = ev["prompt_tokens"]
        elif ev["type"] == "token": text += ev["text"]
        elif ev["type"] == "done": stats.update({k: ev[k] for k in ("tokens", "first_token_ms", "tok_per_s")})
    return dict(prompt=prompt, rag=rag, answer=text.strip(), sources=[s["title"] for s in sources], **stats)
Q = json.load(open(os.path.join(ROOT, "evaluation/results/demo500_selection.json")))
sci = [it for it in Q["Science questions (SciQ test, no passage)"]["shown"] if len(it["a"].split()) <= 3][:2]
sq = Q["Reading comprehension (SQuAD v2, passage given)"]["shown"][0]
allsq = json.load(open(os.path.join(ROOT, "evaluation/results/demo500_all_answers.json")))
miss = [it for it in allsq["Science questions (SciQ test, no passage)"] if not it["correct"] and len(it["q"]) < 90][0]
SQP = "Answer the question using only the context. If the answer is not in the context, say \"I don't know.\"\n\nContext: {c}\n\nQuestion: {q}"
out = dict(science=[dict(ask(it["q"]), gold=it["gold"]) for it in sci], passage=dict(ask(SQP.format(c=sq["ctx"], q=sq["q"])), question=sq["q"], context=sq["ctx"], gold=sq["gold"]),
           miss=dict(ask(miss["q"]), gold=miss["gold"]), rag=[ask(q, rag=True) for q in ("Who wrote the novel Pride and Prejudice?", "What is the capital city of Australia?")],
           norag=[ask(q) for q in ("Who wrote the novel Pride and Prejudice?", "What is the capital city of Australia?")])
json.dump(out, open(os.path.join(ROOT, "evaluation/results/demo500_ui_data.json"), "w"), indent=1); print(json.dumps(out, indent=1)[:3000])
