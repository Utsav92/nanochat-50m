"""Build a demo page of questions the 500M chat model (sft/sci500m_b step 661) ANSWERED CORRECTLY on held-out questions.
Everything shown is a real greedy generation. The page also states the overall accuracy of the same run, because showing only hits is a curated selection.
Correct = the gold answer appears in the response and not in the question (same rule as evaluate_quality.py) AND the response is short (<=15 words) AND the gold answer has >=3 characters, to avoid lenient matches.
  NANOCHAT_BASE_DIR=data/mix_sci500 python scripts/make_demo_500m.py"""
import json, os, random, sys, html
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_quality import Gen, rows, norm, ABSTAIN, SQ_PROMPT
from build_sft_science import rows as srows
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); gen = Gen("sft", "sci500m_b", 661)
def ok(resp, golds, q=""):
    n, nq = norm(resp), norm(q)       # a match whose text is already in the question (the model just echoing it) does not count
    return len(resp.split()) <= 15 and not ABSTAIN.search(resp) and any(len(norm(g)) >= 3 and norm(g) in n and norm(g) not in nq for g in golds)
res = {}
rs = rows("squad_v2_squad_v2_validation_0.parquet"); random.Random(5).shuffle(rs); rs = [r for r in rs if r["answers"]["text"] and len(r["context"]) < 1100][:200]
res["Reading comprehension (SQuAD v2, passage given)"] = [dict(q=r["question"], ctx=r["context"], gold=r["answers"]["text"][0], a=(gen(SQ_PROMPT.format(c=r["context"], q=r["question"]), 40) or "")) for r in rs]
sq = srows("sciq_test")[300:700]
res["Science questions (SciQ test, no passage)"] = [dict(q=r["question"].strip(), gold=r["correct_answer"], a=(gen(r["question"].strip(), 24) or "")) for r in sq]
tv = rows("trivia_qa_rc.nocontext_validation_0.parquet"); random.Random(6).shuffle(tv); tv = tv[:300]
res["Trivia (TriviaQA validation, no passage)"] = [dict(q=r["question"], gold=r["answer"]["aliases"][0], golds=r["answer"]["aliases"], a=(gen(r["question"], 40) or "")) for r in tv]
out = {}
for k, items in res.items():
    for it in items: it["correct"] = ok(it["a"], it.get("golds", [it["gold"]]), it["q"])
    good = [it for it in items if it["correct"]]; out[k] = dict(n=len(items), correct=len(good), shown=good[:10]); print(k, len(good), "/", len(items), flush=True)
json.dump(out, open(os.path.join(ROOT, "evaluation/results/demo500_selection.json"), "w"), indent=1); json.dump(res, open(os.path.join(ROOT, "evaluation/results/demo500_all_answers.json"), "w"))
E = html.escape; parts = []
for k, v in out.items():
    cards = "".join(f'<div class="card"><div class="q">{E(it["q"])}</div>' + (f'<details><summary>passage given to the model</summary><p>{E(it["ctx"])}</p></details>' if "ctx" in it else "") + f'<div class="a"><b>Model:</b> {E(it["a"])}</div><div class="g">Reference answer: {E(it["gold"])}</div></div>' for it in v["shown"])
    parts.append(f'<h2>{E(k)}</h2><p class="rate">Overall on this run: {v["correct"]} of {v["n"]} answered correctly ({100*v["correct"]/v["n"]:.0f}%). Showing up to 10 of the correct ones.</p>{cards}')
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>500M model: correct answers</title>
<style>:root{{--bg:#fafaf7;--fg:#1b1b18;--mut:#6b6b63;--card:#fff;--bd:#dcdcd2;--ok:#1d6b3a;--warn:#7a4b00}}@media(prefers-color-scheme:dark){{:root{{--bg:#151513;--fg:#ececE4;--mut:#a0a095;--card:#1f1f1c;--bd:#33332e;--ok:#6fcf97;--warn:#f0b35a}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}}main{{max-width:760px;margin:0 auto;padding:24px 16px 64px}}h1{{font-size:24px}}h2{{font-size:18px;margin-top:32px}}
.note{{border:1px solid var(--warn);color:var(--warn);border-radius:8px;padding:12px 14px;margin:16px 0}}.rate{{color:var(--mut);font-size:14px}}
.card{{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:12px 14px;margin:10px 0}}.q{{font-weight:600}}.a{{color:var(--ok);margin-top:6px}}.g{{color:var(--mut);font-size:13px}}details{{font-size:13px;color:var(--mut)}}</style></head><body><main>
<h1>500M-parameter model: questions it answered correctly</h1>
<div class="note"><b>This is a selection, not typical behaviour.</b> These are real outputs of the fine-tuned 495M model (greedy decoding), filtered to the ones that match the reference answer.
On held-out questions it is mostly wrong: about 5% on trivia and about 5% on science questions without a passage, and near chance on multiple choice. It does better when a passage is given.
Correct = the reference answer appears in a short response and is not just repeated from the question (still a string match, so a few shown answers may be only partly right).</div>{''.join(parts)}</main></body></html>"""
open(os.path.join(ROOT, "app/demo/demo_500m.html"), "w").write(page); print("wrote app/demo/demo_500m.html")
