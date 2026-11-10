"""Download the first N parquet files (HF auto-converted parquet export) of extra training datasets -> data/raw/extra/<name>_<i>.parquet. Data only.
Licenses (see prepare_datasets.py SOURCES for the audit text): tldr-17 CC-BY-4.0 (Reddit content), stack-exchange-preferences CC-BY-SA-4.0,
cosmopedia Apache-2.0 (synthetic), project_gutenberg public domain (US)."""
import json, os, sys, urllib.request
import subprocess
def fetch(url, path):
    """Resumable download (curl -C -) with retries; urlretrieve fails hard on short reads."""
    for attempt in range(8):
        r = subprocess.run(["curl", "-L", "-sS", "-C", "-", "--retry", "5", "--retry-delay", "5", "-o", path + ".part", url])
        if r.returncode == 0: os.replace(path + ".part", path); return
    raise RuntimeError(f"download failed: {url}")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "extra")
PLAN = {  # name: (dataset, config, split, n_files)
  "reddit_tldr": ("webis/tldr-17", "default", "train", 2),
  "stackexchange": ("HuggingFaceH4/stack-exchange-preferences", "default", "train", 6),
  "gutenberg_en": ("manu/project_gutenberg", "default", "en", 2),
  "cosmo_wikihow": ("HuggingFaceTB/cosmopedia", "wikihow", "train", 1),
  "cosmo_web_v2": ("HuggingFaceTB/cosmopedia", "web_samples_v2", "train", 3),
}
only = sys.argv[1:] or list(PLAN)
for name in only:
    ds, cfg, sp, n = PLAN[name]
    files = [f for f in json.load(urllib.request.urlopen(f"https://datasets-server.huggingface.co/parquet?dataset={ds}", timeout=60))["parquet_files"] if f["config"] == cfg and f["split"] == sp and f["size"] > 5e6]
    for i, f in enumerate(files[:n]):
        p = os.path.join(OUT, f"{name}_{i}.parquet")
        if not os.path.exists(p): fetch(f["url"], p)
        print(name, i, round(os.path.getsize(p) / 1e6), "MB", flush=True)
