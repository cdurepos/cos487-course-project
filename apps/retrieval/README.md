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

The resulting indexes are saved to the `indexes/` directory:

```text
indexes/
├── paragraph_index_unstemmed.json
├── paragraph_index_stemmed.json
├── paper_index_unstemmed.json
└── paper_index_stemmed.json
```

## Retrieval Systems

Retrieval system code and documentation will be added to this directory.
