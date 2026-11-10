# nanochat-50m: small from-scratch language models on a Mac (MPS)

Pipelines built on top of [karpathy/nanochat](https://github.com/karpathy/nanochat) (commit 92d63d4, kept as a separate clone; our changes are in `patches/`):
pretraining (33M/50M/125M/500M-class configs), SFT, small-scale RL, evaluation, RAG, a local preview app and demo videos.

**Status / honesty notes** (details and all numbers in `reports/progress.md`; raw results in `evaluation/results/`):
- Everything was trained from scratch on one Apple Silicon Mac mini (24 GB). No pretrained weights are used.
- The 500M model (494.9M params, ~649M training tokens) does **not** answer questions reliably: SciQ multiple choice 20.7% (chance 25%), PubMedQA = always "yes", TriviaQA 4.7%, passage-based SQuAD 21%.
- Single runs, 300-question evals (about +/-5 points), regex-based abstention detection.
- A second 500M model on a ~1.2B-token education corpus (Khan/OpenStax/Stanford-style Cosmopedia, MIT OCW transcripts, peS2o, Wikipedia, ...) was still training when this was pushed; no results for it yet.
- The demo videos are rendered from recorded real outputs (not screen captures); the Q&A examples in them are selected, and the videos also show the real scoreboard.

**Not in this repo:** datasets, checkpoints, logs (`data/`, `logs/`, `checkpoints/` are git-ignored). Data sources and licences are listed in `scripts/prepare_datasets.py` (`SOURCES`).
Some are research-only (educational Reddit via TLDR-17, SciQ CC-BY-NC, MIT OCW CC BY-NC-SA); Cosmopedia is synthetic (Mixtral) and can contain factual errors. HBS Online and other proprietary course content are not used.
`evaluation/results/*.json` contain a few excerpts of SQuAD (CC BY-SA 4.0) and TriviaQA questions as example model outputs.

Layout: `scripts/` (data prep, training wrappers, evals, demos), `configs/`, `patches/` (changes to upstream), `rag/`, `app/` (preview server + UI), `tests/`, `reports/`, `evaluation/results/`, `video/`.
