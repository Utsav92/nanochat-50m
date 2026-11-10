"""Validation bits-per-byte of checkpoint(s) on the val split of a given data dir, via upstream loader + evaluate_bpb.
Same val dir + same loader => identical batches for every model => comparable numbers.
  NANOCHAT_BASE_DIR=<eval data dir> python scripts/evaluate_bpb.py --ckpt-dir <base>/base_checkpoints/<tag> --steps 100 760"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "upstream"))
import torch
from nanochat.checkpoint_manager import build_model
from nanochat.common import autodetect_device_type
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit as loader
from nanochat.loss_eval import evaluate_bpb
from nanochat.tokenizer import get_token_bytes
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt-dir", required=True); ap.add_argument("--steps", type=int, nargs="+", required=True)
ap.add_argument("--eval-tokens", type=int, default=524288); ap.add_argument("--B", type=int, default=8); ap.add_argument("--T", type=int, default=512)
a = ap.parse_args(); device = torch.device(autodetect_device_type()); tb = get_token_bytes(device=device)
for st in a.steps:
    model, tok, meta = build_model(a.ckpt_dir, st, device, phase="eval")
    n = a.eval_tokens // (a.B * a.T)
    bpb = evaluate_bpb(model, loader(tok, a.B, a.T, split="val", device=device), n, tb)
    print(f"RESULT ckpt={os.path.basename(a.ckpt_dir)} step={st} val_dir={os.environ['NANOCHAT_BASE_DIR']} eval_tokens={n*a.B*a.T} val_bpb={bpb:.4f}", flush=True)
