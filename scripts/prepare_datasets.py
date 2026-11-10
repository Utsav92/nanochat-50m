"""Auditable corpus builder -> nanochat-compatible parquet shards (+ manifest).

Stages per source (counts logged after each): read -> normalize -> empty/length filter -> language heuristic
-> quality filter -> PII redaction -> exact dedup -> near-dup (MinHash-LSH) -> tokenize/budget.
Dedup is GLOBAL (across sources, before the split) so near-duplicates can't straddle train/val.
Split is deterministic: sha1(normalized text) -> val if bucket < val_pct.
Output: <out>/base_data_climbmix/shard_XXXXX.parquet (column `text`, last shard = val, as nanochat expects),
        <out>/tokenizer -> symlink to the frozen tokenizer, data/manifests/dataset_manifest.jsonl.
Usage: NANOCHAT_BASE_DIR=<dir with tokenizer/> python scripts/prepare_datasets.py --profile research
"""
import argparse, hashlib, json, os, random, re, sys, time, unicodedata, zlib
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "upstream"))
from nanochat.tokenizer import get_tokenizer

SOURCES = {  # name: (path, text builder, license, url, revision)
  "fineweb_edu": dict(path="data/raw/fineweb_edu/000_00000.parquet", cols=["text"], license="ODC-By 1.0",
      url="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu", rev="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
      usage="research+commercial with attribution (underlying web content retains original rights)"),
  "wikipedia": dict(path="data/raw/wikipedia/train-00000-of-00041.parquet", cols=["title", "text"], license="CC-BY-SA 3.0 / GFDL",
      url="https://huggingface.co/datasets/wikimedia/wikipedia", rev="b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
      usage="share-alike: derived models/data may carry attribution + SA obligations; review before distribution"),
  # continued-pretraining shards (disjoint from the shards above; downloaded by scripts/download_more_pretrain.sh)
  "fineweb_edu_b": dict(path="data/raw/fineweb_edu/001_00000.parquet", cols=["text"], license="ODC-By 1.0",
      url="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu", rev="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
      usage="research+commercial with attribution (underlying web content retains original rights)"),
  "wikipedia_b": dict(paths=[f"data/raw/wikipedia/train-0000{i}-of-00041.parquet" for i in (1, 2, 3)], path="data/raw/wikipedia/train-00001-of-00041.parquet", cols=["title", "text"], license="CC-BY-SA 3.0 / GFDL",
      url="https://huggingface.co/datasets/wikimedia/wikipedia", rev="b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
      usage="share-alike: derived models/data may carry attribution + SA obligations; review before distribution"),
  # ---- science500 corpus sources (data/raw/extra, data/raw/science; see scripts/download_hf_extra.py, download_science.py) ----
  "pes2o": dict(kind="pes2o", paths=["data/raw/science/pes2o_v2_00000.json.gz"], license="ODC-By 1.0 (underlying papers: open-access licenses via S2ORC/Semantic Scholar)",
      url="https://huggingface.co/datasets/allenai/peS2o", rev="main(v2 shard 0)", usage="attribution; original papers retain their (open-access) licenses; check before redistribution"),
  "wikipedia_all": dict(paths=[f"data/raw/wikipedia/train-0000{i}-of-00041.parquet" for i in range(0, 7)], cols=["title", "text"], license="CC-BY-SA 3.0 / GFDL",
      url="https://huggingface.co/datasets/wikimedia/wikipedia", rev="b04c8d1ceb2f5cd4588862100d08de323dccfbaa", usage="share-alike: derived models/data may carry attribution + SA obligations"),
  "fineweb_edu_all": dict(paths=[f"data/raw/fineweb_edu/{n}.parquet" for n in ("000_00000", "001_00000")], cols=["text"], license="ODC-By 1.0",
      url="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu", rev="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9", usage="attribution (underlying web content retains original rights)"),
  "cosmo_auto_math": dict(paths=["data/raw/science/cosmo_auto_math_0.parquet"], cols=["text"], license="Apache-2.0 (synthetic)",
      url="https://huggingface.co/datasets/HuggingFaceTB/cosmopedia", rev="auto_math_text", usage="permissive; synthetic text may contain factual errors"),
  "cosmo_wikihow": dict(paths=["data/raw/extra/cosmo_wikihow_0.parquet"], cols=["text"], license="Apache-2.0 (synthetic)",
      url="https://huggingface.co/datasets/HuggingFaceTB/cosmopedia", rev="wikihow", usage="permissive; synthetic"),
  "cosmo_web_v2": dict(paths=[f"data/raw/extra/cosmo_web_v2_{i}.parquet" for i in range(3)], cols=["text"], license="Apache-2.0 (synthetic)",
      url="https://huggingface.co/datasets/HuggingFaceTB/cosmopedia", rev="web_samples_v2", usage="permissive; synthetic"),
  "gutenberg_en": dict(kind="gutenberg", paths=[f"data/raw/extra/gutenberg_en_{i}.parquet" for i in range(2)], cols=["text"], license="Public domain (US) per Project Gutenberg; dataset card lists no license",
      url="https://huggingface.co/datasets/manu/project_gutenberg", rev="main", usage="US public domain; Gutenberg license headers may remain in text; verify for non-US use"),
  "stackexchange": dict(kind="stackex", paths=[f"data/raw/extra/stackexchange_{i}.parquet" for i in range(6)], cols=["question", "answers"], license="CC-BY-SA 4.0",
      url="https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences", rev="main", usage="attribution + share-alike (content authors credited in the original dataset)"),
  "reddit_tldr": dict(kind="reddit", paths=[f"data/raw/extra/reddit_tldr_{i}.parquet" for i in range(2)], cols=["subreddit", "body"], license="CC-BY-4.0 (dataset card) -- underlying Reddit user content is subject to Reddit's terms and posters' rights",
      url="https://huggingface.co/datasets/webis/tldr-17", rev="main", usage="research use; do not redistribute text; usernames dropped, PII regex redaction applied, educational subreddits only"),
  "cosmopedia_openstax": dict(path="data/raw/cosmopedia/openstax.parquet", cols=["text"], license="Apache-2.0 (synthetic, generated by Mixtral)",
      url="https://huggingface.co/datasets/HuggingFaceTB/cosmopedia", rev="0ae6ec63f91742bd2d1eaef4f02232c55d719385", usage="permissive; synthetic text may contain factual errors"),
  "cosmopedia_khan": dict(path="data/raw/cosmopedia/khanacademy.parquet", cols=["text"], license="Apache-2.0 (synthetic)",
      url="https://huggingface.co/datasets/HuggingFaceTB/cosmopedia", rev="0ae6ec63f91742bd2d1eaef4f02232c55d719385", usage="permissive; synthetic"),
  "cosmopedia_stanford": dict(path="data/raw/cosmopedia/stanford.parquet", cols=["text"], license="Apache-2.0 (synthetic)",
      url="https://huggingface.co/datasets/HuggingFaceTB/cosmopedia", rev="0ae6ec63f91742bd2d1eaef4f02232c55d719385", usage="permissive; synthetic"),
  "edu_openstax_all": dict(paths=["data/raw/cosmopedia/openstax.parquet", "data/raw/cosmopedia/openstax_1.parquet"], cols=["text"], license="Apache-2.0 (synthetic, generated by Mixtral)",
      url="https://huggingface.co/datasets/HuggingFaceTB/cosmopedia", rev="main", usage="permissive; synthetic text may contain factual errors"),
  "edu_stanford_all": dict(paths=["data/raw/cosmopedia/stanford.parquet"] + [f"data/raw/cosmopedia/stanford_{i}.parquet" for i in range(1, 6)], cols=["text"], license="Apache-2.0 (synthetic)",
      url="https://huggingface.co/datasets/HuggingFaceTB/cosmopedia", rev="main", usage="permissive; synthetic course-style text, may contain factual errors"),
  "mit_ocw": dict(paths=[f"data/raw/mit/ocw_{x}.parquet" for x in ("train", "valid", "test")], cols=["text"], license="MIT OpenCourseWare content is CC BY-NC-SA 4.0 (the HF mirror declares no license): non-commercial research use, share-alike",
      url="https://huggingface.co/datasets/jablonkagroup/mit-ocw-lecture-transcripts", rev="main", usage="research only; real lecture transcripts (physics, chemistry, biology, nuclear); ~11M tokens"),
  "replay_sci500": dict(paths=[f"data/mix_sci500/base_data_climbmix/shard_{i:05d}.parquet" for i in range(1, 78, 3)], cols=["text"], license="mixture: see the sci500 manifest (replay of already-licensed training text; the sci500 validation shard is excluded)",
      url="local", rev="local", usage="replay to limit forgetting during continued pretraining"),
  "finemath": dict(path="data/raw/finemath/finemath-4plus-00000.parquet", cols=["text"], license="ODC-By 1.0 (+ Common Crawl terms)",
      url="https://huggingface.co/datasets/HuggingFaceTB/finemath", rev="e92b25a616738fe95dc186b64dfb19f9c8525594", usage="attribution; underlying web rights apply"),
}
PROFILES = {  # token budgets (kept tokens)
  "smoke":    dict(fineweb_edu=150_000, wikipedia=50_000, synthetic_arithmetic=20_000),
  # Stage 1 (per user plan): 50M tokens, FineWeb-Edu + Wikipedia only
  "research": dict(fineweb_edu=33_000_000, wikipedia=17_000_000),
  # Stage 2 sources registered; token budgets to be set once more shards are downloaded
  # Stage 1b: ~85M-token pool so a ~50M-token run is <=1 pass despite best-fit cropping (~39% at T=512 with 512-token chunks)
  "research85": dict(fineweb_edu=57_000_000, wikipedia=28_000_000),
  # 125M run: ~500M-token pool for ~300M consumed (best-fit cropping factor 1.64 at 512-token chunks)
  "pre125": dict(fineweb_edu=280_000_000, wikipedia=130_000_000, cosmopedia_openstax=20_000_000, cosmopedia_khan=8_000_000,
                 cosmopedia_stanford=20_000_000, finemath=40_000_000),
  # continued pretraining of the 125M model on NEW shards: Wikipedia-heavy (facts) + FineWeb-Edu; ~660M pool for ~400M consumed
  "pre125b": dict(wikipedia_b=380_000_000, fineweb_edu_b=280_000_000),
  # 500M model: science-focused mix. Budgets are kept-token caps (first docs of each source after filtering).
  "sci500": dict(pes2o=230_000_000, wikipedia_all=160_000_000, fineweb_edu_all=100_000_000, finemath=40_000_000, cosmopedia_openstax=20_000_000, cosmopedia_khan=8_000_000,
                 cosmopedia_stanford=20_000_000, cosmo_auto_math=30_000_000, cosmo_wikihow=15_000_000, cosmo_web_v2=30_000_000, gutenberg_en=40_000_000,
                 stackexchange=50_000_000, reddit_tldr=10_000_000),
  # continued pretraining of the finished 500M model: school -> undergraduate material (Cosmopedia OpenStax/Khan/Stanford, ~500M) + ~100M replay of the first corpus
  "edu500": dict(edu_openstax_all=90_000_000, cosmopedia_khan=15_000_000, edu_stanford_all=370_000_000, replay_sci500=100_000_000),
  # 500M model from scratch on a ~1.2B-token pool: sci500 sources + full education sets (Cosmopedia OpenStax/Khan/Stanford) + MIT OCW lecture transcripts
  "edu1b": dict(pes2o=230_000_000, wikipedia_all=160_000_000, fineweb_edu_all=100_000_000, finemath=40_000_000, edu_openstax_all=90_000_000, cosmopedia_khan=15_000_000,
                edu_stanford_all=370_000_000, mit_ocw=12_000_000, cosmo_auto_math=30_000_000, cosmo_wikihow=15_000_000, cosmo_web_v2=30_000_000, gutenberg_en=40_000_000,
                stackexchange=50_000_000, reddit_tldr=10_000_000),
  "stage2_preview": dict(fineweb_edu=20_000_000, wikipedia=10_000_000, cosmopedia_openstax=5_000_000, cosmopedia_khan=3_000_000,
                         cosmopedia_stanford=5_000_000, finemath=5_000_000, synthetic_arithmetic=2_000_000),
}
STOP = set("the of and to in a is that for it as was with be by on not he this are or his from at which but have an had they you were their one all we can her has there been if more when will would who so no".split())
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"); IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
PHONE = re.compile(r"(?<!\d)(?:\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)")
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

def normalize(t):
    t = unicodedata.normalize("NFC", t); t = CTRL.sub("", t).replace("\r\n", "\n")
    return re.sub(r"\n{3,}", "\n\n", t).strip()

def is_english(t):
    s = t[:5000]; letters = sum(c.isalpha() for c in s)
    if letters < 50: return False
    ascii_letters = sum(c.isascii() and c.isalpha() for c in s)
    words = s.lower().split()
    return ascii_letters / letters > 0.9 and sum(w.strip(".,;:!?\"'()") in STOP for w in words) / max(len(words), 1) > 0.15

def quality_ok(t, min_chars=200):
    if len(t) < min_chars: return False
    w = t.split()
    if not w: return False
    if not 3 <= sum(map(len, w)) / len(w) <= 10: return False
    if sum(not (c.isalnum() or c.isspace()) for c in t[:5000]) / min(len(t), 5000) > 0.3: return False
    lines = [l for l in t.split("\n") if l.strip()]
    return not (len(lines) > 5 and len(set(lines)) / len(lines) < 0.5)

def redact_pii(t):
    n = len(EMAIL.findall(t)) + len(IP.findall(t)) + len(PHONE.findall(t))
    if n:
        t = EMAIL.sub("[EMAIL]", t); t = IP.sub("[IP]", t); t = PHONE.sub("[PHONE]", t)
    return t, n

class MinHashLSH:
    def __init__(self, perms=64, bands=16, seed=0):
        r = np.random.RandomState(seed)
        self.a = r.randint(1, 2**31 - 1, perms).astype(np.uint64)[:, None]; self.b = r.randint(0, 2**31 - 1, perms).astype(np.uint64)[:, None]
        self.bands, self.rows = bands, perms // bands; self.tables = [set() for _ in range(bands)]
    def seen_or_add(self, text):
        w = text.lower().split()[:2000]
        if len(w) < 5: return False
        x = np.array([zlib.crc32(" ".join(w[i:i + 5]).encode()) for i in range(len(w) - 4)], dtype=np.uint64)
        sig = ((self.a * x[None, :] + self.b) & np.uint64(0xFFFFFFFF)).min(axis=1)
        keys = [bytes(sig[i * self.rows:(i + 1) * self.rows]) for i in range(self.bands)]
        dup = any(k in t for k, t in zip(keys, self.tables))
        if not dup:
            for k, t in zip(keys, self.tables): t.add(k)
        return dup

def synthetic_arithmetic(n_tokens_est, seed=0):
    """Deterministic arithmetic docs; every answer is computed by Python and re-verified before emission."""
    r = random.Random(seed); out = []; chars = 0
    while chars < n_tokens_est * 3:
        lines = []
        for _ in range(r.randint(8, 20)):
            op = r.choice("+-*/"); a, b = r.randint(0, 10 ** r.randint(1, 3)), r.randint(1, 10 ** r.randint(1, 3))
            if op == "/": b = r.randint(1, 20); a = b * r.randint(0, 50)
            ans = {"+": a + b, "-": a - b, "*": a * b, "/": a // b}[op]
            assert eval(f"{a}{'//' if op == '/' else op}{b}") == ans
            lines.append(f"{a} {op} {b} = {ans}")
        d = "Arithmetic practice with worked answers.\n" + "\n".join(lines); out.append(d); chars += len(d)
    return out

SEG = re.compile(r'.+?(?:[.!?]["\')\]]*(?=\s)|\n|\Z)\s*', re.S)   # sentence/line segments, trailing whitespace kept -> lossless

def chunk_docs(items, tok, limit):
    """Split each (name,text) into chunks of <= `limit` tokens at sentence/line boundaries (lossless concatenation).
    A BOS is prepended to each chunk by the loader, so limit must be <= T (rows are T+1 incl. BOS) to avoid cropping."""
    out = []
    for lo in range(0, len(items), 500):
        batch = items[lo:lo + 500]; segs = [SEG.findall(t) for _, t in batch]
        for (_, t), sg in zip(batch, segs): assert "".join(sg) == t, "segmentation not lossless"
        flat = [x for sg in segs for x in sg]; counts = iter(len(e) for e in tok.encode(flat, num_threads=8))
        for (name, _), sg in zip(batch, segs):
            cur, n = [], 0
            for x in sg:
                c = next(counts)
                if c > limit:                               # pathological single segment: split on words
                    if cur: out.append((name, "".join(cur).strip())); cur, n = [], 0
                    w = x.split(" "); piece = []
                    for word in w:
                        if len(tok.encode(" ".join(piece + [word]))) > limit and piece: out.append((name, " ".join(piece))); piece = []
                        piece.append(word)
                    if piece: cur, n = [" ".join(piece) + " "], len(tok.encode(" ".join(piece)))
                    continue
                if n + c > limit and cur: out.append((name, "".join(cur).strip())); cur, n = [], 0
                cur.append(x); n += c
            if cur: out.append((name, "".join(cur).strip()))
    return [(n, t) for n, t in out if t]

import gzip, html
REDDIT_OK = {"science", "askscience", "explainlikeimfive", "AskHistorians", "history", "math", "physics", "chemistry", "biology", "space", "Astronomy", "technology", "philosophy",
             "books", "todayilearned", "engineering", "programming", "Art", "psychology", "environment", "geology", "medicine", "AskScienceDiscussion", "literature", "Economics", "cogsci", "neuroscience"}
def html_to_text(h):
    h = re.sub(r"<pre[^>]*>\s*<code[^>]*>(.*?)</code>\s*</pre>", lambda m: "\n```\n" + m.group(1) + "\n```\n", h, flags=re.S)
    h = re.sub(r"<code>(.*?)</code>", r"`\1`", h, flags=re.S); h = re.sub(r"<br\s*/?>|</p>|</li>|</h\d>", "\n", h); h = re.sub(r"<[^>]+>", "", h)
    return re.sub(r"\n{3,}", "\n\n", html.unescape(h)).strip()
def source_docs(name, spec):
    kind = spec.get("kind", "text")
    if kind == "pes2o":                                    # JSONL.GZ: {"id","text","source",...}; streamed
        for path in spec["paths"]:
            with gzip.open(os.path.join(ROOT, path), "rt") as f:
                for line in f: yield json.loads(line)["text"]
        return
    for path in (spec["paths"] if "paths" in spec else [spec["path"]]):
        pf = pq.ParquetFile(os.path.join(ROOT, path))
        for b in pf.iter_batches(batch_size=1000, columns=spec["cols"]):   # streaming, never whole file in RAM
            d = b.to_pydict()
            if kind == "reddit":                            # educational subreddits only; body text only (no usernames); PII redaction runs downstream
                for sub, body in zip(d["subreddit"], d["body"]):
                    if sub in REDDIT_OK: yield body
                continue
            if kind == "gutenberg":                         # whole books: keep only the text between the START/END markers (drops the license boilerplate)
                for t in d["text"]:
                    a = re.search(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", t, re.S | re.I); b = re.search(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", t, re.I)
                    yield t[(a.end() if a else 0):(b.start() if b else len(t))].strip()
                continue
            if kind == "stackex":                           # question + best answer (selected, else highest score) as plain text with code fences
                for q, ans in zip(d["question"], d["answers"]):
                    if not ans: continue
                    best = max(ans, key=lambda a: (bool(a.get("selected")), a.get("pm_score") or 0))
                    yield "Question: " + html_to_text(q) + "\n\nAnswer: " + html_to_text(best["text"])
                continue
            for i in range(len(d["text"])):
                yield (f"{d['title'][i]}\n\n{d['text'][i]}" if "title" in d else d["text"][i])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--profile", default="research", choices=PROFILES)
    ap.add_argument("--out", default=None); ap.add_argument("--val-pct", type=float, default=1.0)
    ap.add_argument("--chunk-tokens", type=int, default=0, help="if >0, split docs into <=N-token chunks AFTER the doc-level split (use N<=T-1... e.g. 480 for T=512)")
    ap.add_argument("--shard-docs", type=int, default=20000); ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    out = a.out or os.path.join(ROOT, "data", "mix_" + a.profile); data_dir = os.path.join(out, "base_data_climbmix")
    os.makedirs(data_dir, exist_ok=True); os.makedirs(os.path.join(ROOT, "data/manifests"), exist_ok=True)
    tok_src = os.environ.get("NANOCHAT_BASE_DIR"); assert tok_src and os.path.isdir(os.path.join(tok_src, "tokenizer")), "set NANOCHAT_BASE_DIR to a dir containing tokenizer/"
    tok_link = os.path.join(out, "tokenizer")
    if not os.path.exists(tok_link): os.symlink(os.path.join(os.path.abspath(tok_src), "tokenizer"), tok_link)
    tok = get_tokenizer(); budget = PROFILES[a.profile]
    exact, lsh = set(), MinHashLSH(); docs = []; manifest = []
    for name, tokens_goal in budget.items():
        st = dict(read=0, empty_or_short=0, not_english=0, low_quality=0, pii_docs=0, pii_items=0, exact_dup=0, near_dup=0, kept=0, tokens=0, bytes=0)
        t0 = time.time()
        it = iter(synthetic_arithmetic(tokens_goal)) if name == "synthetic_arithmetic" else source_docs(name, SOURCES[name])
        buf = []
        def flush():
            if not buf: return
            for t, n in zip(buf, tok.encode(buf, num_threads=8)):
                if st["tokens"] >= tokens_goal: break
                st["tokens"] += len(n) + 1; st["kept"] += 1; st["bytes"] += len(t.encode()); docs.append((name, t))
            buf.clear()
        for raw in it:
            st["read"] += 1
            t = normalize(raw)
            if name != "synthetic_arithmetic":
                if len(t) < 200: st["empty_or_short"] += 1; continue
                if not is_english(t): st["not_english"] += 1; continue
                if not quality_ok(t): st["low_quality"] += 1; continue
                t, n = redact_pii(t)
                if n: st["pii_docs"] += 1; st["pii_items"] += n
            h = hashlib.sha1(" ".join(t.lower().split()).encode()).digest()
            if h in exact: st["exact_dup"] += 1; continue
            exact.add(h)
            if name != "synthetic_arithmetic" and lsh.seen_or_add(t): st["near_dup"] += 1; continue
            buf.append(t)
            if len(buf) >= 2000:
                flush()
                if st["tokens"] >= tokens_goal: break
        flush()
        print(f"[{name}] {json.dumps(st)}  ({time.time()-t0:.0f}s)", flush=True)
        spec = SOURCES.get(name, dict(url="generated by scripts/prepare_datasets.py (seeded, verified)", rev="seed=0", license="CC0 (generated)", usage="unrestricted"))
        manifest.append(dict(dataset=name, source_url=spec["url"], revision=spec["rev"], license=spec["license"], language="en",
            documents=st["kept"], bytes=st["bytes"], tokens=st["tokens"], permitted_usage=spec["usage"],
            preprocessing="NFC normalize; len>=200; english heuristic; quality heuristics; PII regex redaction; sha1 exact dedup; MinHash-LSH(64 perms,16 bands,5-word shingles) near dup; global dedup before split",
            stage_counts=st))
    # deterministic split on content hash, deterministic shuffle for mixing
    random.Random(a.seed).shuffle(docs)
    bucket = lambda t: int.from_bytes(hashlib.sha1(t.encode()).digest()[:4], "big") % 10000
    val = [d for d in docs if bucket(d[1]) < a.val_pct * 100]; train = [d for d in docs if bucket(d[1]) >= a.val_pct * 100]
    assert not ({t for _, t in val} & {t for _, t in train})
    if a.chunk_tokens:
        n_docs = (len(train), len(val)); train = chunk_docs(train, tok, a.chunk_tokens); val = chunk_docs(val, tok, a.chunk_tokens)
        vset = {t for _, t in val}; before = len(train); train = [d for d in train if d[1] not in vset]   # identical boilerplate chunks must not straddle the split
        print(f"chunked (<= {a.chunk_tokens} tok): train {n_docs[0]:,} docs -> {before:,} chunks ({before-len(train)} removed as identical to a val chunk), val {n_docs[1]:,} -> {len(val):,}")
        random.Random(a.seed + 1).shuffle(train); random.Random(a.seed + 1).shuffle(val)
    files = []
    def write(path, items):
        pq.write_table(pa.table({"text": [t for _, t in items]}), path + ".tmp", row_group_size=1024); os.replace(path + ".tmp", path); files.append(path)
    for i in range(0, len(train), a.shard_docs): write(os.path.join(data_dir, f"shard_{i // a.shard_docs:05d}.parquet"), train[i:i + a.shard_docs])
    write(os.path.join(data_dir, f"shard_{len(files):05d}.parquet"), val)   # last shard = validation
    for m in manifest:
        m["split_assignment"] = f"train/val by sha1 bucket (val={a.val_pct}%)"
        m["val_documents"] = sum(1 for n, _ in val if n == m["dataset"]); m["train_documents"] = sum(1 for n, _ in train if n == m["dataset"])
    shards = [dict(file=os.path.basename(f), sha256=hashlib.sha256(open(f, "rb").read()).hexdigest(), bytes=os.path.getsize(f)) for f in files]
    with open(os.path.join(ROOT, "data/manifests", f"dataset_manifest_{a.profile}{'_chunked' if a.chunk_tokens else ''}.jsonl"), "w") as f:
        for m in manifest: f.write(json.dumps(m) + "\n")
        f.write(json.dumps(dict(shards=shards, tokenizer_vocab=tok.get_vocab_size(), profile=a.profile, seed=a.seed, chunk_tokens=a.chunk_tokens)) + "\n")
    print(f"train docs {len(train):,}  val docs {len(val):,}  shards {len(files)}  total tokens {sum(m['tokens'] for m in manifest):,}")
    print(f"\nTrain with: NANOCHAT_BASE_DIR={out}")
if __name__ == "__main__": main()
