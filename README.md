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
configs/               Evaluation settings and sparse model paths
bin/                   Install and run scripts
apps/
├── finetuning/        SPLADE fine-tuning
├── indexing/          Sparse and dense index builders
├── retrieval/         Sparse and dense search
├── evaluation/        Run retrieval and score the results
├── utils/             Shared sparse encoding, paths, and index shards
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

Edit `configs/eval.yaml` to change query sets, metrics, levels, and stemming. Sparse checkpoint and index paths are in `configs/sparse.yaml`. See `apps/evaluation/README.md` for more details.

To set up environment and install dependencies:

```bash
bash bin/install.sh
```

To run the research pipeline:

```bash
bash bin/run.sh --eval
```

### Manual Run
Install Python dependencies once from the repository root:

```bash
pip install -r requirements.txt
```

##### Sparse Retrieval
From the repository root, with dependencies installed:

```bash
python -m apps.finetuning.sparse
python -m apps.indexing.sparse
python -m apps.evaluation.evaluate
```

Fine-tuning writes the SPLADE checkpoint. Indexing builds the paragraph and paper indexes from `configs/sparse.yaml`. Evaluation searches those indexes and writes runs and metrics. Details are in the READMEs under `apps/finetuning/`, `apps/indexing/`, `apps/retrieval/`, and `apps/utils/`.

##### Evaluation
```bash
python -m apps.evaluation.evaluate
```

## Course Required Documentation

### Team Member Contributions

#### Clayton Durepos

- Sparse Neural Model Experiment Code: 
`finetuning/sparse.py`, `indexing/sparse.py`, `retrieval/sparse.py` 
- Evaluation Pipeline
`evaluation/evaluate.py`

#### Ella Hawkins

- Indexer Component
- BM25 System
- `install.sh`, `run.sh`