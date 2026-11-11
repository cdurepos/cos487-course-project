# Retrieval

Contains the retrieval code for both sparse and dense models.

## Sparse retrieval

Ranks queries against a sparse SPLADE index. This covers `sparse.py`. Evaluation calls `search()`; there is no separate command.

```python
from apps.retrieval.sparse import search, warmup

warmup(index, level, strategy="maxp")
search(model, level, query, top_k=100, strategy="maxp")
```

`warmup` reads that level's shards into memory and returns nothing. `search` encodes the query, scores those shards, and cosine-normalizes. `level` is `paragraph` or `paper`. For papers, `strategy` is `maxp` or `pool`, and each one loads its own shards. `query` is one string, or a list of `(qid, text)` pairs. The return value is the top `top_k` documents. Evaluation calls `warmup` once, then times each `search` call.

Paragraph search returns body paragraphs and drops ids ending in `_0`. Paper search with `maxp` gives each paper the score of its best passage, including the title and abstract. `pool` scores the precomputed paper vectors.

## Dense retrieval

Contains the `search()` and `delete_models_and_indexes()` functions.

```python
from apps.retrieval.dense import search, delete_models_and_indexes

search(query, biencoder_training, level, paper_method, paragraph_method, crossencoder_training, top_k=100)
delete_models_and_indexes()
```

`search()` encodes the query and searches the specified level. `training` is `pretrained` or `finetuned`, which is applied to the bi-encoder. `level` is `paragraph` or `paper`. For papers, `paper_method` is `mean_pooling`, `from_paragraphs`, or `crossencoder`. For paragraphs, `paragraph_method` is `cosine_similarity` or `crossencoder`. If either method is `crossencoder`, then `crossencoder_training` must be specified (options are `pretrained` or `finetuned`). The return value is the top-k documents and their scores.
