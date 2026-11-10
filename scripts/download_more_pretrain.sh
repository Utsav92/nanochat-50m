#!/bin/bash
# Additional raw shards (data only) for continued pretraining. Pinned revisions as in prepare_datasets.py SOURCES.
cd "$(dirname "$0")/.."
for i in 1 2 3; do n=0000$i; curl -L -sS -C - -o data/raw/wikipedia/train-$n-of-00041.parquet "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/20231101.en/train-$n-of-00041.parquet" || exit 1; done
curl -L -sS -C - -o data/raw/fineweb_edu/001_00000.parquet "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/sample/10BT/001_00000.parquet" || exit 1
echo 0 > logs/dl_more_done.flag
