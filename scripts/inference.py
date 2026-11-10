"""Streaming text completion from a nanochat checkpoint (base or sft) using upstream Engine (KV cache).
NANOCHAT_BASE_DIR selects the data/checkpoint root. Example:
  NANOCHAT_BASE_DIR=data/mix_research python scripts/inference.py --tag stage1_50m -p "The water cycle is" --seed 1 --temperature 0.8 --top-k 50"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "upstream"))
import torch
from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type
from nanochat.engine import Engine

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prompt", required=True); ap.add_argument("--source", default="base", choices=["base", "sft"])
ap.add_argument("--tag", default=None); ap.add_argument("--step", type=int, default=None)
ap.add_argument("--max-tokens", type=int, default=128); ap.add_argument("--temperature", type=float, default=0.8)
ap.add_argument("--top-k", type=int, default=50); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--device", default=None); ap.add_argument("--greedy", action="store_true")
a = ap.parse_args()
device = torch.device(a.device or autodetect_device_type())
model, tok, meta = load_model(a.source, device, phase="eval", model_tag=a.tag, step=a.step)
engine = Engine(model, tok)
ids = [tok.get_bos_token_id()] + tok.encode(a.prompt)
print(a.prompt, end="", flush=True)
t0 = time.time(); first = None; n = 0; out = []
for col, _ in engine.generate(ids, max_tokens=a.max_tokens, temperature=0.0 if a.greedy else a.temperature, top_k=a.top_k, seed=a.seed):
    t = col[0]; n += 1; first = first or time.time() - t0; out.append(t)
    print(tok.decode([t]), end="", flush=True)
dt = time.time() - t0
print(f"\n\n[{n} tokens, first-token {first*1000:.0f} ms, {n/dt:.1f} tok/s, device {device}, step {meta.get('step')}]")
