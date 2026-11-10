# Retrieval

Ranks queries against a sparse SPLADE index. This covers `sparse.py`. Evaluation calls `search()`; there is no separate command.

```python
from apps.retrieval.sparse import search, warmup

warmup(index, level, strategy="maxp")
search(model, level, query, top_k=100, strategy="maxp")
```

`warmup` reads that level's shards into memory and returns nothing. `search` encodes the query, scores those shards, and cosine-normalizes. `level` is `paragraph` or `paper`. For papers, `strategy` is `maxp` or `pool`, and each one loads its own shards. `query` is one string, or a list of `(qid, text)` pairs. The return value is the top `top_k` documents. Evaluation calls `warmup` once, then times each `search` call.

Paragraph search returns body paragraphs and drops ids ending in `_0`. Paper search with `maxp` gives each paper the score of its best passage, including the title and abstract. `pool` scores the precomputed paper vectors.
