"""Replays nanochat's BOS-aligned best-fit packing (nanochat/dataloader.py) on token LENGTHS of the real train shards.
Reports tokens consumed, tokens discarded by cropping, and UNIQUE document tokens ever seen (crops keep a doc's head)."""
import argparse, glob, os, sys
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "upstream"))
from nanochat.tokenizer import get_tokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--T", type=int, nargs="+", default=[512, 1024, 2048])
ap.add_argument("--tokens", type=int, default=49_807_360, help="tokens to consume (default: stage-1 run: 760*65536)")
a = ap.parse_args()
tok = get_tokenizer(); bos = tok.get_bos_token_id()
files = sorted(glob.glob(os.path.join(a.data, "base_data_climbmix", "*.parquet")))[:-1]   # train shards (last = val)
lens = []
for f in files:
    pf = pq.ParquetFile(f)
    for g in range(pf.num_row_groups):
        for t in tok.encode(pf.read_row_group(g).column("text").to_pylist(), prepend=bos, num_threads=8): lens.append(len(t))
lens = np.array(lens); N = len(lens); print(f"train docs {N:,}, tokens {lens.sum():,}, mean {lens.mean():.0f}, median {int(np.median(lens))}, frac docs > 513: {(lens>513).mean():.2f}")
for T in a.T:
    cap = T + 1; rows = a.tokens // T; nxt = 0; epoch = 1
    buf_id, buf_len = [], []                       # same buffer discipline as upstream (refill in batches of 128 until >=1000)
    seen = np.zeros(N, dtype=np.int64); cropped = 0; used = 0; pk = 0
    def refill():
        global nxt, epoch
        for _ in range(128):
            buf_id.append(nxt); buf_len.append(lens[nxt]); nxt += 1
            if nxt == N: nxt = 0; epoch += 1
    for _ in range(rows):
        pos = 0
        while pos < cap:
            while len(buf_id) < 1000: refill()
            rem = cap - pos; bl = np.array(buf_len)
            ok = np.where(bl <= rem, bl, 0); i = int(ok.argmax())
            if ok[i] > 0:
                d = buf_id.pop(i); L = buf_len.pop(i); seen[d] = max(seen[d], L); pos += L; used += L
            else:
                i = int(bl.argmin()); d = buf_id.pop(i); L = buf_len.pop(i)
                seen[d] = max(seen[d], rem); cropped += L - rem; pos += rem; used += rem
    print(f"T={T:5d}: consumed {used:,} tok | discarded by cropping {cropped:,} ({100*cropped/(used+cropped):.1f}% of tokens pulled from docs) | "
          f"unique doc tokens seen {seen.sum():,} ({100*seen.sum()/lens.sum():.1f}% of corpus) | passes over corpus {epoch-1 + nxt/N:.2f} | repeat factor {used/seen.sum():.2f}x")
