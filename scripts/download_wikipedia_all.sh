#!/bin/bash
# Remaining English Wikipedia shards (data only) for the RAG index. Resumable.
cd "$(dirname "$0")/.."
for i in $(seq 0 40); do n=$(printf %05d $i); f=data/raw/wikipedia/train-$n-of-00041.parquet
  if upstream/.venv/bin/python -c "import pyarrow.parquet as pq,sys; pq.ParquetFile(sys.argv[1])" $f 2>/dev/null; then continue; fi
  for t in 1 2 3 4 5; do curl -L -sS -C - -o $f.part "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/20231101.en/train-$n-of-00041.parquet" && mv $f.part $f && break; sleep 5; done
done
echo 0 > logs/dl_wiki_all_done.flag
