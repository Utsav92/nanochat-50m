"""Compute planner. FLOPs use upstream GPT.estimate_flops() (6*matmul params + attention) on the meta model, not just 6*N*D.
Throughput must be MEASURED and passed in (--tok-per-sec); nothing is assumed. Cost scenarios: pass --rate ($/GPU-hour) and --gpus.
Example: python scripts/compute_estimator.py --depth 18 --n-embd 896 --vocab 32768 --seq 2048 --tokens 10e9 --tok-per-sec 400000 --gpus 1 --rate 2.5"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "upstream"))
import torch
from nanochat.gpt import GPT, GPTConfig
ap = argparse.ArgumentParser()
ap.add_argument("--depth", type=int, required=True); ap.add_argument("--n-embd", type=int, required=True)
ap.add_argument("--head-dim", type=int, default=64); ap.add_argument("--vocab", type=int, default=32768)
ap.add_argument("--seq", type=int, default=2048); ap.add_argument("--window", default="SSSL")
ap.add_argument("--tokens", type=float, required=True)
ap.add_argument("--tok-per-sec", type=float, nargs="+", required=True, help="measured throughput (several = low/base/high scenarios)")
ap.add_argument("--gpus", type=int, default=1); ap.add_argument("--rate", type=float, default=None, help="$ per GPU-hour (user supplied)")
a = ap.parse_args()
cfg = GPTConfig(a.seq, a.vocab, a.depth, a.n_embd // a.head_dim, a.n_embd // a.head_dim, a.n_embd, a.window)
with torch.device("meta"): m = GPT(cfg)
N = sum(p.numel() for p in m.parameters()); fpt = m.estimate_flops()
print(f"params {N:,} | FLOPs/token {fpt:.3e} (naive 6N = {6*N:.3e}) | total training FLOPs {fpt*a.tokens:.3e} (naive 6ND = {6*N*a.tokens:.3e})")
print(f"tokens/param {a.tokens/N:.1f} | ckpt fp32 {N*4/1e9:.2f} GB; with Adam-style optimizer state ~{N*12/1e9:.2f} GB")
for tps in a.tok_per_sec:
    secs = a.tokens / tps; h = secs / 3600
    line = f"throughput {tps:,.0f} tok/s -> {h:,.1f} h ({h/24:,.2f} days)"
    if a.rate is not None: line += f" | cost ${h*a.gpus*a.rate:,.0f} at ${a.rate}/GPU-h x {a.gpus}"
    print(line)
