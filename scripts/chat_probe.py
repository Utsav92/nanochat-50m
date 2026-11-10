"""Greedy chat probe: ask a model factual questions and print answers. Usage: chat_probe.py --source rl --tag pre125m --step 75"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "upstream"))
import torch
from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type
from nanochat.engine import Engine
ap = argparse.ArgumentParser(); ap.add_argument("--source", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--step", type=int, default=None); a = ap.parse_args()
Q = ["What is the capital of France?", "Who wrote Romeo and Juliet?", "What is photosynthesis?", "What is 17 + 25?", "What planet is closest to the Sun?",
     "Write a Python function that returns the square of a number.", "Who is the president of the planet Zorblax?", "What is the boiling point of water in Celsius?"]
device = torch.device(autodetect_device_type()); model, tok, meta = load_model(a.source, device, phase="eval", model_tag=a.tag, step=a.step); eng = Engine(model, tok)
for q in Q:
    ids = tok.render_for_completion({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": ""}]})
    out, _ = eng.generate_batch(ids, num_samples=1, max_tokens=80, temperature=0.0)
    print("Q:", q, "\nA:", tok.decode(out[0][len(ids):]).strip()[:300].replace("\n", " "), "\n", flush=True)
