"""Tests for the corpus readers, science SFT formats, answer parsing and the RAG prompt builder (no GPU needed). Run: upstream/.venv/bin/python -m pytest tests -q"""
import json, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "upstream")); sys.path.insert(0, ROOT)
import pytest

def test_html_to_text_keeps_code_fences_and_strips_tags():
    import prepare_datasets as P
    t = P.html_to_text("<p>Use <code>int</code> here.</p><pre><code>x = 1\nprint(x)</code></pre><p>&lt;done&gt;</p>")
    assert "```\nx = 1\nprint(x)\n```" in t and "`int`" in t and "<done>" in t and "<p>" not in t

def test_reddit_reader_filters_subreddits_and_drops_usernames(tmp_path):
    import pyarrow as pa, pyarrow.parquet as pq, prepare_datasets as P
    pq.write_table(pa.table({"subreddit": ["askscience", "funny", "history"], "body": ["a" * 300, "b" * 300, "c" * 300], "author": ["u1", "u2", "u3"]}), tmp_path / "r.parquet")
    spec = dict(kind="reddit", path=str(tmp_path / "r.parquet"), cols=["subreddit", "body"]); P.ROOT = "/"
    assert list(P.source_docs("reddit_tldr", spec)) == ["a" * 300, "c" * 300]

def test_gutenberg_reader_strips_license_boilerplate(tmp_path):
    import pyarrow as pa, pyarrow.parquet as pq, prepare_datasets as P
    book = "Header junk\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nREAL TEXT\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nlicense junk"
    pq.write_table(pa.table({"text": [book]}), tmp_path / "g.parquet"); P.ROOT = "/"
    assert list(P.source_docs("g", dict(kind="gutenberg", path=str(tmp_path / "g.parquet"), cols=["text"]))) == ["REAL TEXT"]

def test_mc_prompt_and_letter_parsing():
    from build_sft_science import mc_prompt
    from evaluate_science import letter_of
    p = mc_prompt("Q?", ["w", "x", "y", "z"]); assert "A. w" in p and "D. z" in p and p.endswith("correct option.")
    assert letter_of("B") == "B" and letter_of(" (C) because") == "C" and letter_of("The answer is B") is None

def test_grounded_prompt_respects_budget_and_template():
    from rag.retriever import grounded_prompt
    ps = [dict(title="A", text="x" * 400), dict(title="B", text="y" * 400), dict(title="C", text="z" * 400)]
    p = grounded_prompt("Who?", ps, max_chars=1000)
    assert p.startswith("Answer the question using only the context.") and p.endswith("Question: Who?") and "C:" not in p and "A:" in p and "B:" in p

def test_abstain_regex_matches_known_phrasings():
    from evaluate_quality import ABSTAIN
    assert all(ABSTAIN.search(s) for s in ["I don't know.", "The context does not mention it.", "I have no information about X"])
    assert not ABSTAIN.search("Paris is the capital of France.")

def test_sft_jsonl_conversations_end_with_assistant():
    d = os.path.join(ROOT, "data", "sft_science")
    if not os.path.isdir(d): pytest.skip("run scripts/build_sft_science.py first")
    for f in os.listdir(d):
        for line in open(os.path.join(d, f)):
            m = json.loads(line)["messages"]; assert m[-1]["role"] == "assistant" and m[-1]["content"].strip() and len(m) % 2 == 0
