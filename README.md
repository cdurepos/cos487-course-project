# COS487 Information Retrieval - Course Project
Search Party's course project for COS487 Information Retrieval at the University of Southern Maine. The repository holds the search interface and the retrieval experiments.

## Team Details
Name: Search Party

Members:
- Clayton Durepos (Contact)
- Ella Hawkins

## Repository Structure
Each part of the project has its own folder under `apps/`.

```text
data/                  Corpus, queries, qrels, indexes, generated replies, metrics, and runs
bin/                   Install and run scripts
apps/
├── processing/        Corpus preprocessing for BM25
├── baseline/          BM25 indexing and search
├── generate.py        Generate and save one reply per query
├── retrieval.py       BM25 search of the original query or a saved reply
├── evaluate.py        Run retrieval and score the results
└── utils.py           Paths and prompts shared by more than one module
```

## Use Guide

### Data Setup
Users must download COS487 Information Retrieval course data files. For installation, ~9GB of disk space is required. Arrange data files as specified below.
```text
data/
├── JSON Files/     Unzipped data corpus
├── qrels/          QREL files
└── queries/        Query files
    ├── Train.json
    ├── Study.json
    └── Test.json
```

### Quick Run
To launch this application on Linux or MacOS, first set up the data (as outlined above). Then, run the following commands in a bash terminal from the repository root.

To set up environment and install dependencies:

```bash
bash bin/install.sh
```

To run retrieval and evaluation:

```bash
bash bin/run.sh
```

### Manual Run
Install Python dependencies once from the repository root:

```bash
pip install -r requirements.txt
```

From the repository root, with dependencies installed:

```bash
python -m src.generate
```
Writes one reply file per query set and prompt under `data/generated/`. Study generation time and token count are written under `data/metrics/generation/`. The run uses seed 42.

```bash
python -m src.processing.preprocess
python -m src.baseline.index
```

Preprocessing writes the cleaned corpus. Indexing builds the stemmed paragraph and paper indexes under `data/indexes/`.

`python src/grid-search.py` fits `k1` and `b` on the Train queries and writes the scores under `data/metrics/grid-search/`.
Best scoring parameters are saved to `src/config.yaml` for use in retrieval & evaluation.

```bash
python -m src.evaluate
```
Searches the original query and the saved replies with BM25 on Study and Test. Study retrieval metrics are written under `data/metrics/`.

## Course Required Documentation

### Team Member Contributions

#### Clayton Durepos

- Evaluation pipeline: `src/evaluate.py`
- Retrieval wrapper: `src/retrieval.py`
- Generation:  `src/generate.py`
- Repo utils: `src/utils.py`

#### Ella Hawkins

- BM25 indexing and retrieval: `src/baseline/index.py`, `src/baseline/bm25.py`
- Grid search: `src/grid-search.py`
- Bash scripts: `bin/install.sh`, `bin/run.sh`
