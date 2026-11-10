import json, os, sys, torch, pytest
ROOT = os.path.join(os.path.dirname(__file__), ".."); sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "upstream"))
from count_parameters import count
RANGES = {"model_50m.yaml": (49_500_000, 50_500_000), "model_500m.yaml": (495_000_000, 505_000_000), "model_125m.yaml": (123_750_000, 126_250_000), "model_33m_baseline.yaml": (32_000_000, 34_000_000)}

@pytest.mark.parametrize("fn", RANGES)
def test_param_count_in_range(fn):
    c = json.load(open(os.path.join(ROOT, "configs", fn)))
    r = count(c["depth"], c["n_embd"], c["head_dim"], c["vocab_size"], c["sequence_len"], c["window_pattern"])
    assert r["total_params"] == c["total_params"]
    assert RANGES[fn][0] <= r["total_params"] <= RANGES[fn][1]

@pytest.mark.parametrize("fn", RANGES)
def test_cli_hint_reproduces_width(fn):  # upstream: dim = ceil(depth*aspect/head_dim)*head_dim
    c = json.load(open(os.path.join(ROOT, "configs", fn))); asp = int(c["cli"].split("aspect-ratio=")[1].split()[0])
    assert -(-c["depth"] * asp // c["head_dim"]) * c["head_dim"] == c["n_embd"]

def test_causal_masking_and_forward_backward():
    from nanochat.gpt import GPT, GPTConfig
    torch.manual_seed(0)
    m = GPT(GPTConfig(64, 256, 2, 2, 2, 128, "L")); m.to_empty(device="cpu"); m.init_weights()
    for p in m.parameters(): p.data.normal_(0, 0.02) if p.dim() > 1 else None   # zero-init c_proj would hide leakage
    x = torch.randint(0, 256, (1, 32)); y = x.clone(); y[0, 20:] = torch.randint(0, 256, (12,))
    with torch.no_grad(): a, b = m(x), m(y)
    assert torch.allclose(a[0, :20], b[0, :20], atol=1e-4)      # future tokens must not change earlier logits
    loss = m(x, targets=x); loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None for p in m.parameters())
