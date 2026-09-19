# Preprocessing — TheSearchParty

Turns the raw arXiv corpus into clean token files that the TF-IDF and BM25
models load directly. Documents and queries go through the same cleaning
function, so their terms always line up.

| | |
|---|---|
| Corpus | 34,308 papers · 2,275,574 paragraphs |
| Queries | 250 Study · 250 Test · 2,000 Train |
| Build time | about 3 minutes |
| Peak memory | about 125 MB |
| Tests | 60 passing (standard library only) |

## Files

| File | Purpose |
|---|---|
| `Preprocessing.py` | Builds the corpus files and provides the loader functions |
| `test_preprocessing.py` | Unit tests |
| `Preprocessing_Documentation.html` | Full write-up: design decisions, defects found, development log |

## Setup

Python 3 only — no packages to install.

Put the course data next to `Preprocessing.py`:

```
JSON Files/          34,308 paper files (not in the repo — too large)
Study.json
Test.json
Train.json
Preprocessing.py
```

## Build

```bash
python Preprocessing.py
```

Writes everything to `processed/`:

| File | Rows | Contents |
|---|---|---|
| `paper_corpus.tsv` | 34,308 | `paper_id <TAB> words` for the whole paper |
| `paragraph_corpus.tsv` | 2,275,574 | `paragraph_id <TAB> words` for one paragraph |
| `paper_titles.tsv` | 34,308 | `paper_id <TAB> title <TAB> abstract`, for the interface |
| `queries_study.tsv` | 250 | `query_id <TAB> words` |
| `queries_test.tsv` | 250 | |
| `queries_train.tsv` | 2,000 | |
| `build_info.txt` | — | Corpus statistics for the report |

For a quick test run, set `LIMIT = 500` at the top of `Preprocessing.py`.

## Use

Load the files through the functions below — don't parse the TSVs yourself.

```python
from Preprocessing import load_corpus, load_queries, load_titles, preprocess

docs    = load_corpus("paper", stem=True)    # or "paragraph"; yields (doc_id, tokens)
queries = load_queries("study", stem=True)   # or "test" / "train"; list of (qid, tokens)
titles  = load_titles()                      # {paper_id: (title, abstract)}

tokens  = preprocess("user typed query", stem=True)   # for live search
```

`load_corpus` is a generator, so build the index in one pass:

```python
index = {}
for doc_id, tokens in load_corpus("paragraph", stem=True):
    for term in tokens:
        index.setdefault(term, []).append(doc_id)
```

### Rules

- **Use the same `stem=` value everywhere.** If documents are stemmed and
  queries aren't, terms stop matching and results get worse with no error.
- **Skip empty documents.** 8,539 paragraphs (0.38%) have no words after
  cleaning. They stay in the file so IDs are complete, but they'll divide by
  zero in length normalization.
- **Normalize for length.** Papers average 3,372 words, paragraphs 51.
- **The search interface must call `preprocess()`** on user input rather than
  cleaning text its own way.

## What the cleaning does

1. Removes LaTeX math (`$x_i$`) and commands (`\alpha`)
2. Removes citations (`[12]`, `[3, 4, 5]`) and URLs
3. Lowercases and splits into words, keeping hyphenated terms like
   `text-to-image` together
4. Drops stopwords and single characters

Saved files are **not stemmed** — stemming happens at load time with
`stem=True`. That keeps the stemming on/off comparison a flag instead of a
rebuild, and lets the interface show real words.

Paper documents include the abstract only if it isn't already the first
paragraph. 87% of papers repeat it, and counting it twice inflated the most
important terms.

## Tests

```bash
python -m unittest test_preprocessing -v
```

## Troubleshooting

**`MemoryError` during the build** — the script only needs about 125 MB, so
this means the machine is out of memory overall. Close browsers and other
heavy programs and rerun.

**Don't run two builds at once** — both write to the same files in
`processed/` and will corrupt each other.

**`FileNotFoundError`** — run the script from the folder that contains
`JSON Files/` and the query JSON files.

## Not in the repo

`JSON Files/`, `processed/` and the other data folders are in `.gitignore`.
They total about 4 GB, over GitHub's 100 MB file limit, and `processed/` can
be rebuilt in a few minutes.
