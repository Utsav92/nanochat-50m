"""Build the BM25 passage index (SQLite FTS5) from raw Wikipedia shards: passages of ~90-130 words split on paragraph boundaries.
  python scripts/build_rag_index.py --shards 0 1 2 --out data/rag/wiki_passages.db
  python scripts/build_rag_index.py --shards $(seq 0 40) --lead 3 --out data/rag/wiki_leads.db   # whole encyclopedia, 3 lead passages per article"""
import argparse, os, re, sqlite3, sys, time
import pyarrow.parquet as pq
ap = argparse.ArgumentParser(); ap.add_argument("--shards", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--out", default="data/rag/wiki_passages.db")
ap.add_argument("--lead", type=int, default=0, help="keep only the first N passages of each article (0 = all)"); ap.add_argument("--max-words", type=int, default=120); ap.add_argument("--min-words", type=int, default=25); a = ap.parse_args()
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); out = os.path.join(ROOT, a.out)
if os.path.exists(out): os.remove(out)
con = sqlite3.connect(out); con.execute("CREATE VIRTUAL TABLE passages USING fts5(title, text, tokenize='porter unicode61')")
def passages(text):
    cur = []; n = 0
    for para in re.split(r"\n+", text):
        w = para.split()
        if not w: continue
        if n + len(w) > a.max_words and n >= a.min_words: yield " ".join(cur); cur, n = [], 0
        while len(w) > a.max_words: yield " ".join(w[:a.max_words]); w = w[a.max_words:]
        cur.append(" ".join(w)); n += len(w)
    if n >= a.min_words: yield " ".join(cur)
t0 = time.time(); total = 0
for s in a.shards:
    pf = pq.ParquetFile(os.path.join(ROOT, "data", "raw", "wikipedia", f"train-{s:05d}-of-00041.parquet"))
    for b in pf.iter_batches(batch_size=2000, columns=["title", "text"]):
        d = b.to_pydict(); rows = [(t, p) for t, x in zip(d["title"], d["text"]) for i, p in enumerate(passages(x)) if not a.lead or i < a.lead]
        con.executemany("INSERT INTO passages(title, text) VALUES (?, ?)", rows); total += len(rows)
    con.commit(); print(f"shard {s}: {total:,} passages so far ({time.time()-t0:.0f}s)", flush=True)
con.execute("INSERT INTO passages(passages) VALUES('optimize')"); con.execute("CREATE TABLE leads AS SELECT lower(title) AS t, min(rowid) AS rid FROM passages GROUP BY title"); con.execute("CREATE INDEX leads_t ON leads(t)"); con.commit(); print("done", total, "passages", round(os.path.getsize(out) / 1e9, 2), "GB")
