"""Find nanochat GPT configs whose real parameter count is nearest 50M (49.5M-50.5M).
Uses the actual upstream GPT class on the meta device via count_parameters.count().
Unlike a depth-only sweep it also varies vocab size, since embeddings dominate nanochat's count
(wte + untied lm_head + per-layer value embeddings)."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from count_parameters import count, show

ap = argparse.ArgumentParser()
ap.add_argument("--target", type=float, default=50e6)
ap.add_argument("--tol", type=float, default=0.5e6)
ap.add_argument("--seq-len", type=int, default=1024)
ap.add_argument("--vocabs", type=int, nargs="+", default=[16384, 24576, 32768])
ap.add_argument("--top", type=int, default=15)
ap.add_argument("--out", default=None)
a = ap.parse_args()

cands = []
for vocab in a.vocabs:
    for hd in (32, 64):
        for depth in range(4, 9):
            for n_embd in range(256, 513, 32):
                if n_embd % hd == 0 and 4 <= n_embd // hd <= 16:
                    cands.append(count(depth, n_embd, hd, vocab, a.seq_len))
cands.sort(key=lambda r: abs(r["total_params"] - a.target))
print(f"{'vocab':>6} {'depth':>5} {'dim':>4} {'hd':>3} {'heads':>5} {'total':>12} {'non-emb':>10} {'in-range':>8}")
for r in cands[:a.top]:
    ok = abs(r["total_params"] - a.target) <= a.tol
    print(f"{r['vocab_size']:6d} {r['depth']:5d} {r['n_embd']:4d} {r['head_dim']:3d} {r['n_head']:5d} {r['total_params']:12,} {r['non_embedding_params']:10,} {'YES' if ok else 'no':>8}")
best = cands[0]
print("\nClosest:"); show(best); print("Upstream CLI:", best["cli"])
if abs(best["total_params"] - a.target) > a.tol:
    sys.exit(f"VALIDATION FAILED: {best['total_params']:,} outside {a.target:,.0f}±{a.tol:,.0f}")
if a.out:
    json.dump(best, open(a.out, "w"), indent=2); print("saved", a.out)
