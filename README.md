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
data/                  Corpus, queries, qrels, indexes, and runs
bin/                   Install and run scripts
apps/
├── processing/        Corpus preprocessing for BM25
├── baseline/          BM25 indexing and search
├── retrieval.py       Gemma prompt, then BM25 search
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

#### BM25
From the repository root, with dependencies installed:

```bash
python -m apps.processing.preprocess
python -m apps.baseline.index
```

Preprocessing writes the cleaned corpus. Indexing builds the stemmed and unstemmed paragraph and paper indexes under `data/indexes/`. Search uses the stemmed indexes and the parameters in `apps/baseline/config.yaml`.

`python apps/baseline/grid-search.py` fits `k1` and `b` on the Train queries and writes the scores under `data/evaluation/`.

#### Evaluation
```bash
python -m apps.evaluate
```

Runs the original query and every prompt in `PROMPT_MAP` through BM25 on Study and Test. Study is scored against the qrels.

## Course Required Documentation

### Team Member Contributions

#### Clayton Durepos

- Evaluation pipeline: `evaluate.py`

#### Ella Hawkins

- BM25 indexing and retrieval: `baseline/index.py`, `baseline/bm25.py`
- `install.sh`, `run.sh`
