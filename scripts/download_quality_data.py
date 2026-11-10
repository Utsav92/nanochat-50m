"""Download eval/training datasets (data only, no pretrained models). Licenses: SQuAD v2 CC-BY-SA-4.0, HumanEval MIT, MBPP CC-BY-4.0, TriviaQA Apache-2.0."""
import json, os, urllib.request
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "quality")
WANT = {"rajpurkar/squad_v2": [("squad_v2", "train"), ("squad_v2", "validation")], "openai/openai_humaneval": [("openai_humaneval", "test")],
        "google-research-datasets/mbpp": [("sanitized", "train"), ("sanitized", "test"), ("sanitized", "validation"), ("sanitized", "prompt")],
        "mandarjoshi/trivia_qa": [("rc.nocontext", "train"), ("rc.nocontext", "validation")]}
for ds, splits in WANT.items():
    meta = json.load(urllib.request.urlopen(f"https://datasets-server.huggingface.co/parquet?dataset={ds}", timeout=60))["parquet_files"]
    for cfg, sp in splits:
        fs = [f for f in meta if f["config"] == cfg and f["split"] == sp]
        for i, f in enumerate(fs[:2]):
            p = os.path.join(OUT, f"{ds.split('/')[-1]}_{cfg}_{sp}_{i}.parquet")
            if not os.path.exists(p): urllib.request.urlretrieve(f["url"], p)
            print(p, os.path.getsize(p))
