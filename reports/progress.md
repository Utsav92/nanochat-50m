# NanoChat-50M progress (measured facts only)

- Upstream: karpathy/nanochat @ 92d63d4e8bb4df75c3b71618f31ddde2378b2bcd, branch `nanochat-50m` (no upstream edits so far)
- Env: macOS 27.0, arm64, 24 GB, Python 3.10.22 (uv venv), torch 2.9.1, MPS available = True
- Install: `uv sync --extra cpu` (`pip install -e .` fails: setuptools package discovery)
- Upstream pretraining data is ClimbMix-400B (not FineWeb-Edu); NANOCHAT_BASE_DIR controls data/ckpt location
  (currently ./data/nanochat_base on internal disk; no external SSD mounted)
- Param count (real GPT class, meta device): embeddings dominate (wte + untied lm_head + value_embeds on alternate layers).
  - Selected 50M: depth 7, dim 384, 6 heads (hd 64), vocab 16384, seq 1024 -> 50,135,368 params (non-embedding 12,386,632)
    CLI: --depth=7 --aspect-ratio=54 --head-dim=64 (aspect-ratio is int; 7*54=378 -> rounded up to 384)
  - 33M baseline: depth 6, dim 320, 5 heads, vocab 16384 -> 33,587,418 (configs/model_33m_baseline.yaml)
- Tokenizer: 16384 vocab trained on 300M chars of 5 shards (5.8 s). 24576/32768 comparison NOT yet done.
- Smoke run (MPS, fp32/default dtype, seq 512, dev-batch 4, total batch 8192, 30 steps, eager):
  loss 9.704 -> 8.478; val bpb 3.179 -> 2.480; ~28,000 tok/s steady-state (30-step sample only).
  Upstream "peak memory" is CUDA-only (prints 0.00 on MPS) -> own measurement still TODO.
- Resume test (step 15 -> 30): works; losses match original within ~0.002 but not bit-identical (dataloader position differs).
- Naive extrapolation at 28K tok/s: 50M tok ~30 min, 1B tok ~10 h. UNVERIFIED over longer runs.

## Stage 1 result (50M, 760 steps x 65,536 tok, T=512, MPS, 24.9 min, ~32.7K tok/s, ~1.2 GB RSS)
- val bpb: 3.196 (0) -> 1.775 (100) -> 1.554 -> 1.405 -> 1.341 -> 1.314 (500) -> **1.310 (600, best)** -> 1.3155 (700) -> 1.3156 (760)
- Generation ("The water cycle is", seed 1, T=0.8, top-k 50): degenerate repetition ("cycle cycle ..."). 1 prompt = smoke test, not an eval.
- **Data repetition finding** (scripts/measure_cropping.py, replay of upstream best-fit packing on real shards):
  T=512: 69.2% of pulled tokens cropped; only 13.6M unique tokens (27.5% of corpus) seen, 3.7 passes. T=1024: 2.4x; T=2048: 1.8x.
  Matches log `epoch: 4`. => "50M tokens" = tokens consumed, NOT unique. Plan budgets in unique tokens.

## Stage 1b: chunked / larger-pool retrain (50M, identical hyperparameters, only data changed)
- Corpus `data/mix_research85`: 85.0M tokens (57.0M FineWeb-Edu + 28.0M Wikipedia), docs split at sentence boundaries into <=512-token chunks AFTER the
  doc-level hash split (203,983 train / 2,078 val chunks). Replay (scripts/measure_cropping.py): 0.96 passes, repeat factor 1.00x (stage 1: 3.7x).
  Chunking alone did not remove cropping (39% at 512-tok chunks, 22% at 256): best-fit still crops to fill row gaps. A bigger pool does.
- 760 steps x 65,536 tok (49.8M consumed, all unique), 24.94 min, ~32.7K tok/s.
- Like-for-like val bpb (scripts/evaluate_bpb.py, identical batches, 524,288 eval tokens):
  | val set | stage 1 @760 (best @600) | stage 1b @760 | delta |
  | whole-doc (340 docs, stage-1 format) | 1.3156 (1.3100) | **1.2476** | -0.068 (-5.2%) |
  | chunked (2,078 chunks)               | 1.4374 (1.4281) | **1.3125** | -0.125 (-8.7%) |
  Stage 1 got worse after step 600 on both sets; stage 1b improved monotonically. Single seed per run; noise not estimated.
- Generation (3 prompts x 3 seeds, T=0.8, top-k 50, eyeballed): both fluent-looking and factually wrong (no "Paris"). Greedy loops in both.
  Stage 1 collapsed onto "cycle cycle ..." for seed 1/greedy; stage 1b showed no single-token collapse in sampled runs. Modest, qualitative.

## 500M smoke tests on this Mac (scripts/smoke_model.py; 496,371,238 params, random init, B=batch, 1 repeated batch, lr x0.05)
| dtype | B x T | step time | tok/s | allocated | driver-allocated |
|---|---|---|---|---|---|
| fp32 | 1x512  | 1.06 s | ~485   | 5.27 GB | 12.3 GB |
| fp32 | 1x2048 | 1.98 s | ~1,030 | 5.27 GB | 18.5 GB |
| bf16 | 1x512  | 0.45 s | ~1,130 | 3.97 GB | 9.2 GB |
| bf16 | 1x2048 | 1.10 s | ~1,855 | 3.99 GB | 16.2 GB |
| bf16 | 1x2048, window SSSL | 1.12 s | ~1,820 | 3.99 GB | 16.1 GB |
| bf16 | 2x2048 | 7 -> 20 -> 45 s | 571 -> 91 | 3.97 GB | **25.8 GB (> 24 GB physical; thrashing)** |
- Loss falls 10.40 -> 8.94 in 10 fp32 steps (memorizing one batch: optimizer-update check, NOT generalization). Initial loss 10.40 = ln(32768).
- bf16 is ~1.7-2.3x faster and ~25% less memory than fp32 on MPS here, loss trajectory matches fp32 closely over the few steps tested.
  Numerical stability over a long run is UNTESTED. torch.compile on MPS untested.
- Sliding-window (SSSL) gives no speedup on MPS (SDPA fallback masks). B=2 at T=2048 exceeds RAM => local 500M training capped at B=1.
- Rough extrapolation at ~1,855 tok/s (bf16, B=1): 100M tokens ~15 h, 10B tokens ~62 days. From a 3-step measurement; indicative only.
  => Full 500M pretraining belongs on cloud GPUs (no cloud throughput/cost measured yet; none assumed).
- Harness note: first B=2 crash was a non-contiguous `.view` in my smoke script (fixed), not an upstream bug.

## App preview
- `app/backend/server.py` (FastAPI, SSE streaming via upstream Engine) + `app/frontend/index.html` (single file). Base-model text continuation only; no chat/SFT yet.
  Run: `NANOCHAT_BASE_DIR=data/mix_research85 upstream/.venv/bin/python -m uvicorn app.backend.server:app --port 8000`
  Info panel is read from checkpoint metadata. Seed-1 output over HTTP matched the CLI output token-for-token. Layout not visually verified by me.

## Process notes / mistakes caught
- A background wait loop using `pgrep -f` matched its own command line and hung 35 min; results were re-run directly. Avoid that pattern.
- Several zsh loops failed from unsplit variables (not script bugs). GSM8K "last number" grader gave credit to a degenerate loop -> added accidental-match baseline.

## 50M SFT + RL (base = stage1b_50m)
- SFT (upstream chat_sft; mixture SmolTalk 460K + MMLU-aux x3 + GSM8K-train x4, 600 steps x 65,536 tok, 20 min). Val bpb on the SFT mixture 1.265 -> 0.636.
  **Patch 0001**: upstream's SFT packer deadlocked into all-padding batches (NaN loss) at 512 context because conversations longer than a row are never
  evicted from its buffer; patch skips them (those long conversations are therefore never trained on; count not measured). Also: upstream `--num-iterations` counts micro-batches.
- RL (patch 0002: `--task arithmetic`, tasks/arithmetic.py; train/val disjoint by hash; 1-2 digit +,-,small x). Reward saturated at 1.0 by step ~49 -> stopped at step 75/500.
  Val pass@1 (100 held-out problems, T=1): step0 2% -> 25: 28% -> 50: 74% -> 75: 66% (pass@8 9% -> 100% -> 100% -> 93%).
- **What RL learned: to call the calculator tool** (`<|python_start|>expr<|python_end|>`, executed by the Engine). Arithmetic accuracy is the model+tool system.
  | checkpoint | arith 2-digit held-out (n=300) tool ON | tool OFF (network alone) | 3-digit (n=300) tool ON | tool OFF |
  | SFT 600 | 7 (2.3%) | 7 (2.3%) | 1 (0.3%) | 0 |
  | RL 50 | **300 (100%)** | 5 (1.7%, all small x) | 288 (96.0%) | 0 |
  | RL 75 | 300 (100%) | 5 (1.7%) | 293 (97.7%) | 0 |
  The network cannot add (0/250 + and - tool-off). Model still often fails to stop after answering (loops); reward ignored it.
- Knowledge benchmarks (chat letter-format MC, n=2376/1172): SFT ARC-Easy 25.9% (chance 25.0%), ARC-Challenge 30.4%. **Letter bias**: predicts B 44-47%, A ~3%;
  best constant-letter baseline 26.5% => no demonstrated MC ability. RL50: ARC-E 25.8%, ARC-C 29.6%. GSM8K (n=200): SFT 1.0%, RL50 0.5% (accidental level).
- Base-model raw log-likelihood ARC-Easy was 39.1% (stage1b) - a different protocol; the SFT'd model did NOT improve on knowledge MC. SFT mainly fixed format/behavior.

## 125M model
- Config: depth 9, dim 672, 7 heads (head_dim 96), vocab 16384 -> 125,841,872 params (non-embedding 48.8M). `configs/model_125m.yaml`.
  (32K-vocab alternative in range had only 25M non-embedding params.)
- Real-trainer benchmark (B=8x512, 65,536 tok/step, 10 steps): fp32 11.1K tok/s; **bf16 27.8K tok/s** (2.5x); loss at step 9 9.546 vs 9.548. ~6 GB driver memory.
  Time at 27.8K tok/s (extrapolated): 300M tok ~3 h, 1B ~10 h, 2.5B (20 tok/param) ~25 h.
- Chosen: ~300M tokens consumed (2.4 tok/param: far below Chinchilla), from a ~500M-token unique pool (cropping factor ~1.64 at 512-token chunks).

---
## 125M pipeline (pre125m): pretraining, SFT, RL, quality evals

### Pretraining (measured)
depth 9 / n_embd 672 / 7 heads / vocab 16384 / T=512 (trained at 512), 4,580 steps x 65,536 tok (~300M tokens) on data/mix_pre125, bf16, 178.6 min, ~27.8K tok/s. Val bpb 3.496 -> 0.695. No NaNs.
Base eval: ARC-Easy 36.6% (log-lik, chance 25%), ARC-Challenge 20.5% raw / 23.3% norm (chance), arithmetic ~0%, GSM8K 1.5% (accidental baseline 2%).

### Run v1 (SFT 600 steps -> arithmetic RL 75 steps) -- evaluation/results/pre125m_{sft600,rl75}.json
2-digit held-out arithmetic: tool on 93.7%, tool off 1.7%. 3-digit: tool on 61%, tool off 0%. ARC (letters) at chance, GSM8K 1.5%.
=> RL taught calculator-tool use; the network does not calculate.

### Quality evals added (scripts/evaluate_quality.py; greedy; single seed; n=300 unless stated)
Data: SQuAD v2 validation (held out), TriviaQA validation, HumanEval (164), MBPP sanitized test (257), 60 fake-entity questions (hand-templated).
| checkpoint | fake-entity halluc. | SQuAD-unans. halluc. | SQuAD-ans. correct | SQuAD over-abstain | TriviaQA correct (abstain) | HumanEval | MBPP |
|---|---|---|---|---|---|---|---|
| v1 RL75 (baseline) | 100% | 64% | 13.7% | 35% | 6.7% (0.3%) | 0/164 | 4/257 (1.6%) |
| v2 SFT (7.5K SQuAD) | 60% | 29% | 10.0% | 62% | 4.0% (56%) | - | - |
| v2 RL, *unshaped* reward (COLLAPSED) | 23% | 2% | 1.0% | 98% | 0.7% (90%) | - | - |
| v3 SFT (45K SQuAD) | 60% | 38% | 15.7% (F1 .18) | 54% | 3.7% (59%) | 0/164 | 0/257 |
| v3 RL75 (shaped reward) | 5% | 14% | 6.7% | 75% | 1.0% (91%) | 0/164 (22% define fn) | 0/257 |
v3 RL75 arithmetic (tool on/off): 2-digit 33%/1.7%, 3-digit 69%/0%.

### What went wrong / honest findings
1. RL v2 reward hacking: first mixed reward let "I don't know" on *answerable* questions cost nothing; an all-abstain policy already scored ~87% of max reward. The policy collapsed to abstaining everywhere; its reported 85% Pass@1 and "2% hallucination" were NOT real improvements. Result kept as evaluation/results/q125_rl_v2_collapsed.json. Fixed by shaping the reward (patch 0004) and rebalancing the mix.
2. Shaped RL still trades usefulness for caution: hallucination falls (fake 100%->5%, unanswerable 64%->14%) but correct answers fall too (SQuAD answerable 13.7%->6.7%; TriviaQA 6.7%->1%). Abstention only weakly tracks answerability (75% vs 86%), i.e. the 125M model's reading comprehension/knowledge, not the training recipe, is the bottleneck.
3. Code did not improve: HumanEval 0/164 in every checkpoint; MBPP 1.6% -> 0% after the v3 SFT mix (SmolTalk cut to 30K rows). The v3 SFT checkpoint (before RL) already has MBPP 0, so the regression is from the data mix, not RL. A code-capable model needs code in pretraining (the corpus has none).
4. Closed-book knowledge (TriviaQA 4-7%) is capacity/data-limited (~300M training tokens); SFT/RL cannot add facts. More pretraining tokens/parameters is the real lever.
5. Operational: SFT v2 stopped at step 343 (one epoch of a 92.8K-row mixture, not 600) and saved under the same tag -- my first v2 evaluation silently loaded the old v1 step-600 checkpoint (caught because the numbers were implausibly identical); checkpoints are now separated by tag (pre125m, pre125m_v2, pre125m_v3). RL on MPS hit driver OOM at ~step 29-100 in each long run (patched with periodic torch.mps.empty_cache + device batch 4; still crashes ~step 75-100); runs were stopped at the last saved checkpoint.
6. Caveats: single seeds; 300-example evals (CI roughly +/-5 pts); abstention is detected by regex; the fake-entity train templates/names are disjoint from the eval ones but the fake-entity eval is small and templated; no n-gram decontamination against the web pretraining text; RL leg boundaries (restart from step-25 checkpoint) reset the optimizer and example order in the collapsed run.

### Patches
0003 (SFT extra JSONL + row caps) and 0004 (mixed RL task + MPS cache clearing) are cumulative diffs of the touched files (they include 0001/0002 changes in chat_sft.py / chat_rl.py).

---
## 500M science model (sci500m) -- plan, data, status  [pretraining IN PROGRESS at time of writing]

### Model
depth 22, width 1024, 16 heads (head_dim 64), vocab 16,384 (the same frozen tokenizer as the 50M/125M models), T=1024, window "L": **494,930,054 parameters** (non-embedding 276.8M) -- configs/model_500m_v16k.yaml.
The earlier 32K-vocab 500M config (configs/model_500m.yaml) was NOT used: it needs a new tokenizer and would make 125M-vs-500M comparisons confounded. Consequence: scientific terminology is split into more tokens than a science-trained tokenizer would (not measured).
Smoke test (bf16, B=2, T=1024): 15.6 GB driver memory, finite losses. Measured during real training: ~4,900 tok/s, ~13.3 s/step (65,536 tokens/step) => 9,900 steps = 649M tokens ~ 37 h.

### Pretraining corpus (data/mix_sci500; scripts/prepare_datasets.py --profile sci500 --chunk-tokens 1024)
| source | kept docs | tokens | license / notes |
|---|---|---|---|
| peS2o v2 shard 0 (allenai) | 760,385 | 230.0M | ODC-By; mostly paper ABSTRACTS (~300 tok/doc), not full text; first 230M tokens of the shard |
| Wikipedia (shards 0-6) | 124,409 | 160.0M | CC-BY-SA 3.0/GFDL; first docs of the shards |
| FineWeb-Edu (2 shards) | 88,273 | 100.0M | ODC-By |
| FineMath-4plus (shard 0) | 24,633 | 40.0M | ODC-By |
| Cosmopedia: openstax / khan / stanford | 23,706 / 8,350 / 20,047 | 20.0M / 8.0M / 20.0M | Apache-2.0, synthetic (can contain factual errors) |
| Cosmopedia: auto_math_text / wikihow / web_samples_v2 | 41,184 / 15,579 / 38,554 | 30.0M / 15.0M / 30.0M | Apache-2.0, synthetic |
| Project Gutenberg (en, 2 files) | 349 books | 40.0M | US public domain (card lists no license); START/END boilerplate stripped; 859 duplicate books removed |
| Stack Overflow Q&A (HF stack-exchange-preferences, 6 files) | 120,138 | 50.0M | CC-BY-SA 4.0; question + best answer, HTML -> text with code fences (this is the only code in pretraining) |
| Reddit TL;DR-17 (webis), educational subreddits only | 12,593 | 4.2M | card: CC-BY-4.0; underlying Reddit content subject to Reddit terms/posters' rights -> research use, usernames dropped, PII regex applied. Allowlist leaves only ~3% of rows |
Pipeline per source: NFC normalize, length>=200, English heuristic, quality heuristics, PII regex redaction (emails/IPs/phones), exact + MinHash-LSH near dedup (global), hash-based train/val split BEFORE chunking, chunks <=1024 tokens.
Totals: train 1,547,485 chunks / 739.6M tokens, val 15,903 chunks; 747.2M tokens in 79 shards. Packing replay at T=1024: a 400M-token run uses 54% of the corpus with 1.00x repeat and 10.9% cropping loss.
NOT used from your list, and why: PubChem, ChEMBL, UniProt, MoleculeNet, QMugs, Materials Project, CERN Open Data, NASA Exoplanet Archive, Ensembl, Reactome, PDB, OpenNeuro, ZINC, Gaia, NOAA, NASA Earthdata, Met/AIC/Smithsonian/Europeana, Wikidata, SEP: these are structured databases (or need per-source access/terms review) and need record-to-text conversion with provenance; not done. PMC-OA/Europe PMC full text: the HF `pmc/open_access` dataset mixes licences including NC/ND and needs the commercial-use subset filter; not done (peS2o abstracts/papers cover similar ground). arXiv: only `armanc/scientific_papers` (licence "unknown") and `ccdv/arxiv-summarization` (no licence) were found, so not used. NuminaMath repo id not found; math is covered by FineMath, Cosmopedia auto_math_text and GSM8K. Wikisource not downloaded.
SciQ is CC-BY-NC-3.0: used for SFT train + held-out evaluation in this research setting only.

### Fine-tuning / evaluation data (data only)
SFT: SmolTalk (40K rows), MMLU auxiliary (15K), GSM8K x2, SQuAD-v2 grounded QA (45K) + abstention, fake-entity abstention (1.5K), MBPP train (170 x5), SciQ train (33.8K conversations: MC, closed-book, passage-grounded), MedQA train (10.2K MC), PubMedQA artificial (11K yes/no/maybe), TriviaQA knowledge-aware labels (from SFT-A).
Held out (never trained on): SciQ test, PubMedQA pqa_labeled (1000), MedQA test, SQuAD v2 validation, TriviaQA validation, HumanEval, MBPP test. RL monitoring uses the tail of the TRAIN tables.

### RAG (scripts/build_rag_index.py, rag/retriever.py; server option "answer from Wikipedia passages")
SQLite FTS5 BM25 (no embeddings, CPU, offline), title-aware re-ranking, grounded prompt in the same template as the SFT QA data. First index covered only shards 0-2 (462K of ~6.4M articles = 7%): well-known articles ("Pride and Prejudice", "Mona Lisa", "photosynthesis") were missing, so retrieval returned unrelated passages. Fix in progress: download all 41 shards (data only) and index lead passages. RAG answer quality has NOT been measured yet; retrieval recall on TriviaQA will be reported when the full index exists.

### Honest expectations
Scale: ~650M tokens for a 495M-parameter model (1.3 tokens/param) is far below compute-optimal (~20 tokens/param). Expect fluent scientific prose, modest QA accuracy (near chance on MedQA/PubMedQA is plausible), and no reliable code or factual recall. Results will be filled in from evaluation/results/*500*.json by scripts/pipeline_500m.sh; nothing here is a result yet.

## 500M sci500m: base + SFT results (measured; greedy; n=300 unless stated; single seed; 95% Wilson CIs in evaluation/results/*.json)
Base (evaluation/results/sci500m_base.json, step 9900, val bpb 0.6007): ARC-Easy 34.6% (chance 25%), ARC-Challenge 21.9% (chance 25%), arithmetic 0-2%, GSM8K 0.5%.
SFT answer-always recipe (sci500m_b step 661; sci500_sft.json, q500_sft.json). The SFT mix was exhausted after 661 steps; SFT val bpb rose from 0.39 (step 0) to 0.45.
- SciQ MC 20.7% (chance 25%) -- predicts letter C 293/300 times (no knowledge; position bias). SciQ closed-book contains gold 5.3%.
- PubMedQA 54.7% = always "yes" = majority baseline (no skill). MedQA 27.7% (chance 25%; CI 23-33%), predicts C 298/300.
- TriviaQA contains-gold 4.7% (accidental baseline 0.3%); abstain rate 0 (answer-always by design). Fake-entity questions: hallucination rate 100%.
- SQuAD v2 (context given): answerable contains-gold 21.3%, F1 0.25, over-abstention 25.3%; unanswerable abstain 40.7%.
- HumanEval pass@1 0.6% (1/164), MBPP 1.2% (3/257).
Conclusion: the SFT'd sci500m model does NOT answer questions correctly. It follows the format but the multiple-choice heads are degenerate (one letter) and factual recall is ~5%.
Bug note: an earlier science eval run returned 0% everywhere because Gen returned None for max_tokens<8; fixed (max_tokens=8) and rerun; the numbers above are from the fixed run.
RL (science task, answer-always) crashed at step ~46 with "probability tensor contains inf/nan" (bf16 sampling); only the step-25 checkpoint exists; no RL results.
