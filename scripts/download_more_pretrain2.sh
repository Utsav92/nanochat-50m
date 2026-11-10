#!/bin/bash
# Second batch of raw shards (data only) for the 500M run's larger unique pool. Waits for batch 1.
cd "$(dirname "$0")/.."
while [ ! -f logs/dl_more_done.flag ]; do sleep 20; done
for i in 4 5 6 7 8; do n=0000$i; curl -L -sS -C - -o data/raw/wikipedia/train-$n-of-00041.parquet "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/20231101.en/train-$n-of-00041.parquet" || exit 1; done
curl -L -sS -C - -o data/raw/fineweb_edu/002_00000.parquet "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/sample/10BT/002_00000.parquet" || exit 1
echo 0 > logs/dl_more2_done.flag
