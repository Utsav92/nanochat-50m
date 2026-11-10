"""Search real nanochat GPT configs nearest 125M params (495M-505M), counted on the meta device."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from count_parameters import count, show

ap = argparse.ArgumentParser()
ap.add_argument("--target", type=float, default=125e6); ap.add_argument("--tol", type=float, default=1.25e6)
ap.add_argument("--seq-len", type=int, default=1024); ap.add_argument("--window-pattern", default="L")
ap.add_argument("--vocabs", type=int, nargs="+", default=[16384, 24576, 32768]); ap.add_argument("--top", type=int, default=15)
ap.add_argument("--out", default=None); a = ap.parse_args()

cands = []
for vocab in a.vocabs:
    for hd in (64, 128):
        for depth in range(8, 17):
            for n_embd in range(512, 1025, 64):
                if n_embd % hd == 0:
                    cands.append(count(depth, n_embd, hd, vocab, a.seq_len, a.window_pattern))
cands.sort(key=lambda r: abs(r["total_params"] - a.target))
print(f"{'vocab':>6} {'depth':>5} {'dim':>5} {'hd':>4} {'heads':>5} {'total':>13} {'non-emb':>12} {'ok':>3}")
for r in cands[:a.top]:
    print(f"{r['vocab_size']:6d} {r['depth']:5d} {r['n_embd']:5d} {r['head_dim']:4d} {r['n_head']:5d} {r['total_params']:13,} {r['non_embedding_params']:12,} {'Y' if abs(r['total_params']-a.target)<=a.tol else 'n':>3}")
best = cands[0]; show(best)
P = best["total_params"]
# training-memory estimate (weights+grads+2 Adam moments, fp32 = 16 B/param) EXCLUDING activations; Muon keeps 1 buffer for matrices, so this is an upper bound
best["est_train_mem_GB_excl_activations_adamw_fp32"] = P * 16 / 1e9
print(f"Est. training state (fp32 AdamW, no activations): {P*16/1e9:.1f} GB; inference fp32 {P*4/1e9:.2f} GB, bf16 {P*2/1e9:.2f} GB")
print("Upstream CLI:", best["cli"])
if abs(P - a.target) > a.tol: sys.exit(f"VALIDATION FAILED: {P:,}")
if a.out: json.dump(best, open(a.out, "w"), indent=2); print("saved", a.out)
