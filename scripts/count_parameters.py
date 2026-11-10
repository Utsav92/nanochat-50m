"""Search for a nanochat GPT config near a target parameter count.

Builds the *real* upstream nanochat.gpt.GPT on the meta device (no memory), counts unique
trainable params (via .parameters(), which de-duplicates shared tensors), and breaks them down.

Usage:
  python scripts/count_parameters.py --target 33e6 --vocab 16384 --out configs/architecture.yaml
  python scripts/count_parameters.py --depth 6 --n-embd 384 --head-dim 64 --vocab 16384   # inspect one
"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "upstream"))
import torch
from nanochat.gpt import GPT, GPTConfig

TOL = (32_000_000, 34_000_000)

def count(depth, n_embd, head_dim, vocab, seq_len, window_pattern="L"):
    assert n_embd % head_dim == 0
    n_head = n_embd // head_dim
    cfg = GPTConfig(sequence_len=seq_len, vocab_size=vocab, n_layer=depth, n_head=n_head,
                    n_kv_head=n_head, n_embd=n_embd, window_pattern=window_pattern)
    with torch.device("meta"):
        m = GPT(cfg)
    uniq = {id(p): p for p in m.parameters()}  # de-duplicate tied tensors
    total = sum(p.numel() for p in uniq.values())
    trainable = sum(p.numel() for p in uniq.values() if p.requires_grad)
    sc = m.num_scaling_params()
    attn = sum(p.numel() for b in m.transformer.h for p in b.attn.parameters())
    mlp = sum(p.numel() for b in m.transformer.h for p in b.mlp.parameters())
    assert sc["total"] == total
    padded_vocab = m.transformer.wte.weight.shape[0]
    return dict(
        depth=depth, n_embd=n_embd, n_head=n_head, head_dim=head_dim, vocab_size=vocab,
        padded_vocab_size=padded_vocab, sequence_len=seq_len, window_pattern=window_pattern,
        total_params=total, trainable_params=trainable,
        breakdown=dict(wte=sc["wte"], lm_head=sc["lm_head"], value_embeds=sc["value_embeds"],
                       attention=attn, mlp=mlp, scalars=sc["scalars"]),
        non_embedding_params=total - sc["wte"] - sc["value_embeds"] - sc["lm_head"],
        mem_fp32_MB=total * 4 / 1e6, mem_bf16_MB=total * 2 / 1e6,
        # reproduce with upstream CLI: model_dim = ceil(depth*aspect/head_dim)*head_dim
        cli=f"--depth={depth} --aspect-ratio={n_embd//depth} --head-dim={head_dim} --max-seq-len={seq_len}",
    )

def show(r):
    b = r["breakdown"]
    print(f"Total params:        {r['total_params']:,}")
    print(f"Trainable params:    {r['trainable_params']:,}")
    print(f"Non-embedding:       {r['non_embedding_params']:,}")
    print(f"Depth / dim / heads: {r['depth']} / {r['n_embd']} / {r['n_head']} (head_dim {r['head_dim']})")
    print(f"Vocab (padded):      {r['vocab_size']} ({r['padded_vocab_size']})")
    print(f"Context length:      {r['sequence_len']}")
    print(f"Breakdown:           " + ", ".join(f"{k}={v:,}" for k, v in b.items()))
    print(f"Weights memory:      fp32 {r['mem_fp32_MB']:.1f} MB, bf16/fp16 {r['mem_bf16_MB']:.1f} MB (excl. grads/optimizer/activations)")

def validate(r, tol=TOL):
    if not (tol[0] <= r["total_params"] <= tol[1]):
        raise SystemExit(f"VALIDATION FAILED: {r['total_params']:,} not in {tol}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=33e6)
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--head-dims", type=int, nargs="+", default=[32, 64])
    ap.add_argument("--depth", type=int); ap.add_argument("--n-embd", type=int); ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--no-validate", action="store_true")
    a = ap.parse_args()
    if a.depth:
        r = count(a.depth, a.n_embd, a.head_dim, a.vocab, a.seq_len); show(r)
        if not a.no_validate: validate(r)
        sys.exit()
    cands = []
    for hd in a.head_dims:
        for depth in range(4, 9):
            for n_embd in range(256, 513, hd if hd >= 64 else 32):
                if n_embd % hd: continue
                cands.append(count(depth, n_embd, hd, a.vocab, a.seq_len))
    cands.sort(key=lambda r: (abs(r["total_params"] - a.target), r["depth"]))
    print(f"{'depth':>5} {'dim':>4} {'hd':>3} {'heads':>5} {'total':>12} {'non-emb':>11} {'embed%':>6}")
    for r in cands[:a.top]:
        print(f"{r['depth']:5d} {r['n_embd']:4d} {r['head_dim']:3d} {r['n_head']:5d} {r['total_params']:12,} {r['non_embedding_params']:11,} {100*(1-r['non_embedding_params']/r['total_params']):5.1f}%")
    best = cands[0]; print("\nClosest:"); show(best)
    if not a.no_validate: validate(best)
    if a.out:
        with open(a.out, "w") as f: f.write(json.dumps(best, indent=2))
        print("saved", a.out)
