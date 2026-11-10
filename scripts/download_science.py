"""Science corpus + QA downloads (data only). peS2o v2 shard 0 (ODC-By), PubMedQA (MIT), MedQA-USMLE (CC-BY-4.0), SciQ (CC-BY-NC-3.0: research/eval use only), Cosmopedia auto_math_text (Apache-2.0)."""
import json, os, urllib.request
import subprocess
def fetch(url, path):
    """Resumable download (curl -C -) with retries; urlretrieve fails hard on short reads."""
    for attempt in range(8):
        r = subprocess.run(["curl", "-L", "-sS", "-C", "-", "--retry", "5", "--retry-delay", "5", "-o", path + ".part", url])
        if r.returncode == 0: os.replace(path + ".part", path); return
    raise RuntimeError(f"download failed: {url}")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "science"); os.makedirs(OUT, exist_ok=True)
def get(url, name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p): fetch(url, p)
    print(name, round(os.path.getsize(p) / 1e6, 1), "MB", flush=True)
PARQ = {"pubmedqa_labeled": ("qiaojin/PubMedQA", "pqa_labeled", "train"), "pubmedqa_artificial": ("qiaojin/PubMedQA", "pqa_artificial", "train"),
        "medqa_train": ("GBaker/MedQA-USMLE-4-options", "default", "train"), "medqa_test": ("GBaker/MedQA-USMLE-4-options", "default", "test"),
        "sciq_train": ("allenai/sciq", "default", "train"), "sciq_test": ("allenai/sciq", "default", "test"), "sciq_validation": ("allenai/sciq", "default", "validation")}
for name, (ds, cfg, sp) in PARQ.items():
    fs = [f for f in json.load(urllib.request.urlopen(f"https://datasets-server.huggingface.co/parquet?dataset={ds}", timeout=60))["parquet_files"] if f["config"] == cfg and f["split"] == sp]
    if fs: get(fs[0]["url"], name + ".parquet")
    else: print("MISSING", name)
fs = [f for f in json.load(urllib.request.urlopen("https://datasets-server.huggingface.co/parquet?dataset=HuggingFaceTB/cosmopedia", timeout=60))["parquet_files"] if f["config"] == "auto_math_text"]
get(fs[0]["url"], "cosmo_auto_math_0.parquet")
get("https://huggingface.co/datasets/allenai/peS2o/resolve/main/data/v2/train-00000-of-00020.json.gz", "pes2o_v2_00000.json.gz")
