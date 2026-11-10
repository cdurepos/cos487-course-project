# Indexing

Builds the sparse indexes that retrieval searches. This covers `sparse.py`.

## Run

From the repository root, after fine-tuning:

```bash
python -m apps.indexing.sparse
```

`--batch-size` defaults to 8. `--interval` defaults to 100 papers saved per shard.

The script indexes every model in `configs/sparse.yaml`. For each paper it encodes the title and abstract as `{paper_id}_0`, encodes each body paragraph, and stores one paper vector: the element-wise max of those passage vectors. Text longer than 256 tokens is chunked before that max is taken.

Indexes are written under the `index` path in `configs/sparse.yaml` and can resume. A changed checkpoint, corpus, token limit, or interval does not resume an existing index.
