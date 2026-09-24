# Retrieval

This directory contains the retrieval components.

## Indexing

`index.py` builds inverted indexes for the preprocessed corpus at both the paragraph and paper levels, with and without stemming.

The indexes contain:

* **Document lengths** — the number of tokens in each document.
* **Term frequencies** — the number of times each term occurs in each document.

### Building the Indexes

Run the following command from the repository root:

```bash
python -m apps.retrieval.index
```

This builds four indexes:

* Paragraph index without stemming
* Paragraph index with stemming
* Paper index without stemming
* Paper index with stemming

The resulting indexes are saved under `data/indexes/`:

```text
data/indexes/
├── paragraph_index_unstemmed.json
├── paragraph_index_stemmed.json
├── paper_index_unstemmed.json
└── paper_index_stemmed.json
```

## Retrieval Systems

There are two retrieval systems implemented: TF-IDF and BM25.

### TF-IDF

`tfidf.py` implements TF-IDF retrieval (vector space model, ranked by cosine similarity) over the indexes built by `index.py`. No extra build step is needed.

The model uses:

* **Log-scaled term frequency** — `1 + ln(tf)`
* **Inverse document frequency** — `ln(N / df)`
* **Cosine similarity** between the query and document vectors, so scores fall between 0 and 1

In SMART notation this is `ltc.ltc`.

The system supports:

* **Paragraph- and paper-level retrieval**
* **Stemmed and unstemmed indexes**
* **Configurable number of results (`k`)**
* **Index and document-norm caching**, so each index is loaded and its document norms are computed only once per process

The `retrieve()` function has the following interface:

```python
retrieve(query, level, stem, k)
```

where:

* `query` is the search query.
* `level` is either `"paragraph"` or `"paper"`.
* `stem` determines whether to use the stemmed or unstemmed index.
* `k` is the number of results to return.

Results are returned as a dictionary mapping document IDs to cosine similarity scores, sorted in descending order of score. This is the same format as BM25, so the two can be swapped.

**Note:** the first search on each index computes document norms, which takes a while on the paragraph index (roughly a minute or two). Later searches are fast.

#### Using TF-IDF

To use the TF-IDF retrieval system interactively, run the following command from the repository root:

```bash
python -m apps.retrieval.tfidf
```

The REPL prompts for:

1. A query
2. The index level (`paragraph` or `paper`)
3. Whether to use stemming
4. The number of results to return

### BM25

`bm25_system.py` implements BM25 retrieval over the indexes built by `index.py`.

The system supports:

* **Paragraph- and paper-level retrieval**
* **Stemmed and unstemmed indexes**
* **Configurable number of results (`k`)**
* **Configurable BM25 parameters** (`k_1` and `b`)
* **Index caching**, so each index only needs to be loaded once per process

The `retrieve()` function has the following interface:

```python
retrieve(query, level, stem, k, k_1=1.2, b=0.75)
```

where:

* `query` is the search query.
* `level` is either `"paragraph"` or `"paper"`.
* `stem` determines whether to use the stemmed or unstemmed index.
* `k` is the number of results to return.
* `k_1` and `b` are the BM25 parameters.

Results are returned as a dictionary mapping document IDs to BM25 scores, sorted in descending order of score.

#### Using BM25

To use the BM25 retrieval system interactively, run the following command from the repository root:

```bash
python -m apps.retrieval.bm25_system
```

The REPL prompts for:

1. A query
2. The index level (`paragraph` or `paper`)
3. Whether to use stemming
4. The number of results to return
