# Preprocessing

Builds clean token files from the arXiv corpus for TF-IDF and BM25. Documents
and queries use the same cleaning function so their terms line up.

| | |
|---|---|
| Files | `preprocess.py`, `test.py` |
| Corpus | 34,308 papers · 2,275,574 paragraphs |
| Dependencies | Python 3 standard library only |

## Setup

Course data lives at the repository root (see the root `README.md`):

```text
data/
├── JSON Files/     Paper corpus
├── Study.json      Queries (also Test.json / Train.json when available)
└── processed/      Written by this script
```

## Build

From the repository root:

```bash
python -m apps.processing.preprocess
```

For a short smoke run, set `LIMIT = 500` near the top of `preprocess.py`.

### Outputs (`data/processed/`)

| File | Contents |
|---|---|
| `paper_corpus.tsv` | `paper_id <TAB> tokens` |
| `paragraph_corpus.tsv` | `paragraph_id <TAB> tokens` |
| `paper_titles.tsv` | `paper_id <TAB> title <TAB> abstract` |
| `queries_study.tsv` | `query_id <TAB> tokens` (same for test/train when present) |
| `build_info.txt` | Corpus statistics |

Saved tokens are cleaned but **not stemmed**. Stemming is applied at load time.

## Loading

```python
from apps.processing.preprocess import load_corpus, load_queries, load_titles, preprocess

docs    = load_corpus("paper", stem=True)     # or "paragraph"; yields (doc_id, tokens)
queries = load_queries("study", stem=True)    # or "test" / "train"
titles  = load_titles()                       # {paper_id: (title, abstract)}
tokens  = preprocess("user query", stem=True) # live queries
```

Use the same `stem=` value for documents and queries. `load_corpus` is a
generator — build indexes in one pass.

Skip empty documents when indexing: some paragraphs have no tokens after
cleaning and will break length normalization.

## Cleaning

1. Strip LaTeX math and commands, citations (`[12]`), and URLs
2. Lowercase and split; keep hyphenated terms such as `text-to-image`
3. Drop stopwords and single-character tokens
4. Deduplicate the abstract when it already appears as a paragraph (~87% of papers)

## Tests

From `apps/processing`:

```bash
python -m unittest test -v
```
