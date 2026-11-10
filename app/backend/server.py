"""Local preview server for NanoChat checkpoints: base models (text continuation) and chat models (SFT / RL, rendered as a user/assistant conversation).
Run: NANOCHAT_BASE_DIR=data/mix_research85 upstream/.venv/bin/python -m uvicorn app.backend.server:app --port 8000   (from repo root)
Streams tokens over Server-Sent Events using upstream Engine (KV cache). Everything shown in /api/info comes from checkpoint metadata."""
import glob, json, os, sys, threading, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "upstream"))
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from nanochat.checkpoint_manager import build_model
from nanochat.common import autodetect_device_type
from nanochat.engine import Engine
sys.path.insert(0, ROOT)

app = FastAPI(title="NanoChat preview"); device = torch.device(autodetect_device_type()); _cache, _lock = {}, threading.Lock()

KINDS = {"base_checkpoints": "base", "chatsft_checkpoints": "sft", "chatrl_checkpoints": "rl"}
HIDE = ("collapsed", "_part1", "v2rl25", "pre125m_v1")   # the reward-hacked RL runs (always answer "I don't know") are not shown

def checkpoints():
    out = {}
    for sub, kind in KINDS.items():
        for d in sorted(glob.glob(os.path.join(ROOT, "data", "*", sub, "*"))):
            tag = os.path.basename(d)
            if any(h in tag for h in HIDE) or os.path.islink(d): continue
            steps = sorted(int(os.path.basename(f)[6:12]) for f in glob.glob(os.path.join(d, "model_*.pt")))
            if steps: out[f"{kind}/{tag}"] = dict(dir=d, last_step=steps[-1], kind=kind, name=tag, chat=kind != "base")
    return out

def load(tag):
    with _lock:
        if tag not in _cache:
            ck = checkpoints().get(tag)
            if not ck: raise HTTPException(404, f"unknown checkpoint {tag}")
            model, tok, meta = build_model(ck["dir"], ck["last_step"], device, phase="eval")
            _cache[tag] = (Engine(model, tok), tok, meta, sum(p.numel() for p in model.parameters()))
        return _cache[tag]

@app.get("/")
def index(): return FileResponse(os.path.join(ROOT, "app", "frontend", "index.html"))

@app.get("/api/checkpoints")
def list_ckpts(): return [dict(tag=t, step=v["last_step"], chat=v["chat"]) for t, v in checkpoints().items()]

@app.get("/api/info/{tag:path}")
def info(tag: str):
    _, tok, meta, n = load(tag); uc = meta.get("user_config", {}); mc = meta["model_config"]; ck = checkpoints()[tag]; step = meta.get("step", ck["last_step"])
    return dict(tag=tag, chat=ck["chat"], parameters=n, step=step, tokens_consumed=step * uc.get("total_batch_size", 0) if not ck["chat"] else None,
                val_bpb=meta.get("val_bpb"), model_config=mc, vocab_size=tok.get_vocab_size(), device=str(device),
                total_batch_size=uc.get("total_batch_size"), max_seq_len=mc["sequence_len"])

class Req(BaseModel):
    tag: str; prompt: str = Field(max_length=4000); rag: bool = False; rag_k: int = Field(2, ge=1, le=4); max_tokens: int = Field(60, ge=1, le=400)
    temperature: float = Field(0.8, ge=0, le=2); top_k: int = Field(50, ge=0, le=1000); seed: int = 42

@app.post("/api/generate")
def generate(r: Req):
    engine, tok, meta, _ = load(r.tag); T = meta["model_config"]["sequence_len"]
    chat = checkpoints()[r.tag]["chat"]; sources = []
    if r.rag and chat:                                         # retrieval-augmented: BM25 over local Wikipedia passages, grounded prompt (same template as SFT)
        from rag.retriever import Retriever, grounded_prompt
        global _retr
        if "_retr" not in globals() or _retr is None: _retr = Retriever()
        sources = _retr.search(r.prompt, k=r.rag_k); r = r.model_copy(update=dict(prompt=grounded_prompt(r.prompt, sources, 1000 if T <= 512 else 2400)))
    ids = tok.render_for_completion({"messages": [{"role": "user", "content": r.prompt}, {"role": "assistant", "content": ""}]}) if chat else [tok.get_bos_token_id()] + tok.encode(r.prompt)
    keep = max(T - r.max_tokens, 1); truncated = len(ids) > keep; ids = ids[-keep:]       # never exceed the trained context length
    max_new = min(r.max_tokens, T - len(ids))
    def stream():
        t0 = time.time(); first = None; n = 0
        yield f"data: {json.dumps(dict(type='start', prompt_tokens=len(ids), truncated=truncated, sources=[dict(title=x['title'], text=x['text'][:300]) for x in sources]))}\n\n"
        with _lock:                                           # one generation at a time on the GPU
            for col, _ in engine.generate(ids, max_tokens=max_new, temperature=r.temperature, top_k=r.top_k or None, seed=r.seed):
                n += 1; first = first or time.time() - t0
                if col[0] in (tok.encode_special("<|assistant_end|>"),): break
                yield f"data: {json.dumps(dict(type='token', text=tok.decode([col[0]])))}\n\n"
        dt = time.time() - t0
        yield f"data: {json.dumps(dict(type='done', tokens=n, first_token_ms=round((first or 0)*1000), tok_per_s=round(n/dt,1)))}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")
