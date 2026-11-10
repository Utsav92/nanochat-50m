"""Forward/backward/optimizer smoke test + memory/step-time measurement of a nanochat GPT on this device, random init.
  python scripts/smoke_model.py --config configs/model_500m.yaml --B 1 --T 512 --steps 3 [--window L]"""
import argparse, json, os, sys, time, resource
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "upstream"))
import torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.common import autodetect_device_type, COMPUTE_DTYPE
ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--B", type=int, default=1); ap.add_argument("--T", type=int, default=512)
ap.add_argument("--steps", type=int, default=3); ap.add_argument("--window", default=None); ap.add_argument("--lr-scale", type=float, default=1.0); a = ap.parse_args()
c = json.load(open(a.config)); dev = torch.device(autodetect_device_type())
cfg = GPTConfig(sequence_len=max(a.T, 512), vocab_size=c["vocab_size"], n_layer=c["depth"], n_head=c["n_head"], n_kv_head=c["n_head"], n_embd=c["n_embd"], window_pattern=a.window or c["window_pattern"])
with torch.device("meta"): m = GPT(cfg)
m.to_empty(device=dev); m.init_weights(); m.train()
n = sum(p.numel() for p in m.parameters()); print(f"device {dev} | compute dtype {COMPUTE_DTYPE} | params {n:,} | window {cfg.window_pattern} | B={a.B} T={a.T}")
opt = m.setup_optimizer()
for gr in opt.param_groups: gr["lr"] *= a.lr_scale
sync = torch.mps.synchronize if dev.type == "mps" else (lambda: None)
mem = (lambda: (torch.mps.current_allocated_memory() / 1e9, torch.mps.driver_allocated_memory() / 1e9)) if dev.type == "mps" else (lambda: (0, 0))
print("after init: allocated %.2f GB, driver %.2f GB" % mem())
g = torch.Generator().manual_seed(0); x = torch.randint(0, c["vocab_size"], (a.B, a.T + 1), generator=g).to(dev)
peak_drv = 0
for s in range(a.steps):
    sync(); t0 = time.time()
    loss = m(x[:, :-1].contiguous(), x[:, 1:].contiguous()); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1e9).item()
    opt.step(); m.zero_grad(set_to_none=True); sync(); dt = time.time() - t0
    al, dr = mem(); peak_drv = max(peak_drv, dr)
    print(f"step {s}: loss {loss.item():.4f} gradnorm {gn:.3f} time {dt:.2f}s ({a.B*a.T/dt:,.0f} tok/s) | allocated {al:.2f} GB driver {dr:.2f} GB | finite={bool(torch.isfinite(loss))}")
print(f"peak driver-allocated {peak_drv:.2f} GB | process max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9:.2f} GB")
