# Utils

Shared pieces for sparse fine-tuning, indexing, and retrieval.

`catalog()` in `sparse.py` reads `configs/sparse.yaml` and returns each model's checkpoint and index path. Fine-tuning, indexing, and evaluation all use it, so those paths stay in one place.

`encode()` in `encoding.py` turns text into one SPLADE vector. Text longer than the token limit is split into chunks, and the chunk vectors are combined with an element-wise max. Sparse indexing, retrieval, and fine-tuning call it with a limit of 256 tokens.

`sharding.py` stores the paragraph and paper vectors as resumable compressed sparse shards, then loads those shards at search time.
