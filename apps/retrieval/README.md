# Retrieval

Ranks queries against a sparse SPLADE index. This covers `sparse.py`. Evaluation calls `search()`; there is no separate command.

```python
from apps.retrieval.sparse import search

search(model, index, level, query, top_k=100, strategy="maxp", timed=False)
```

`level` is `paragraph` or `paper`. `query` is one string, or a list of `(qid, text)` pairs. Scores are cosine similarity, and the return value is the top `top_k` documents.

Paragraph search returns body paragraphs and drops ids ending in `_0`. Paper search with `maxp` gives each paper the score of its best passage, including the title and abstract. `pool` scores the precomputed paper vectors. `timed=True` also returns query latency and index size for evaluation.
