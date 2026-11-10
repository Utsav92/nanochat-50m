"""Local BM25 retriever over Wikipedia passages (SQLite FTS5; no embedding model, no network). Passages carry their article title for attribution.
Wikipedia text is CC-BY-SA 3.0: show the title/source whenever retrieved text is displayed."""
import os, re, sqlite3
_D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "rag")
DB = os.path.join(_D, "wiki_leads.db") if os.path.exists(os.path.join(_D, "wiki_leads.db")) else os.path.join(_D, "wiki_passages.db")   # full-encyclopedia lead index preferred
STOP = set("a an the of and or to in on at by for with from is are was were be been being as it its this that these those which who whom what when where why how do does did has have had not no but if then than so such into about over under between also can could may might will would should".split())

def tokens(q): return [w for w in re.findall(r"[A-Za-z0-9]+", q.lower()) if w not in STOP and len(w) > 1]

class Retriever:
    def __init__(self, db=DB): self.con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, check_same_thread=False)
    def search(self, query, k=3, title_boost=2.0):
        ts = tokens(query)[:24]
        if not ts: return []
        match = " OR ".join(f'"{t}"' for t in dict.fromkeys(ts))
        # bm25 weights: title column boosted; lower score = better in FTS5
        rows = self.con.execute("SELECT rowid, title, text, bm25(passages, ?, 1.0) AS s FROM passages WHERE passages MATCH ? ORDER BY s LIMIT ?", (title_boost, match, max(k * 12, 40))).fetchall()
        qt = re.findall(r"[a-z0-9]+", query.lower()); ql = " " + " ".join(qt) + " "; out = []
        seen = {r[0] for r in rows}
        for n in range(min(5, len(qt)), 0, -1):                                 # articles whose title is an n-gram of the question: add their lead passages as candidates
            for i in range(len(qt) - n + 1):
                g = " ".join(qt[i:i + n])
                if n == 1 and (g in STOP or len(g) < 3): continue
                hit = self.con.execute("SELECT rid FROM leads WHERE t = ?", (g,)).fetchone()
                if hit:
                    for r in self.con.execute("SELECT rowid, title, text, 0.0 FROM passages WHERE rowid IN (?, ?)", (hit[0], hit[0] + 1)).fetchall():
                        if r[0] not in seen: rows.append(r); seen.add(r[0])
        for rid, t, x, s in rows:
            tl = " ".join(re.findall(r"[a-z0-9]+", t.lower())); sc = -s
            if tl and f" {tl} " in ql: sc += 12.0 + len(tl.split())           # article title appears verbatim in the question
            if re.match(r"(list of|lists of|index of|outline of)\b|\(disambiguation\)|soundtrack|discography|filmography", t.lower()): sc -= 6.0
            out.append((sc, rid, t, x))
        out.sort(key=lambda z: (-z[0], z[1]))                                  # ties: earlier passage (article lead) first
        return [dict(title=t, text=x, score=sc) for sc, _, t, x in out[:k]]

def grounded_prompt(question, passages, max_chars=1000):
    """Same template the SFT data used (scripts/build_sft_quality.py SQ_PROMPT), so the model sees a familiar format."""
    ctx = ""
    for p in passages:
        piece = f"{p['title']}: {p['text']}"
        if len(ctx) + len(piece) + 1 > max_chars: break
        ctx += (" " if ctx else "") + piece
    return "Answer the question using only the context. If the answer is not in the context, say \"I don't know.\"\n\nContext: " + ctx + "\n\nQuestion: " + question
