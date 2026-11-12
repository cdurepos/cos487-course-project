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
├── utils/             Sparse encoding, paths, and index shards
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

Edit `configs/eval.yaml` to change query sets, metrics, and levels. Sparse checkpoint and index paths are in `configs/sparse.yaml`. See `apps/evaluation/README.md` for more details.

To set up environment and install dependencies:

```bash
bash bin/install.sh
```

To run the research pipeline:

```bash
bash bin/run.sh
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
```

Fine-tuning writes the SPLADE checkpoint. Indexing builds the paragraph and paper indexes from `configs/sparse.yaml`. Details are in the READMEs under `apps/finetuning/`, `apps/indexing/`, `apps/retrieval/`, and `apps/utils/`.

##### Dense Retrieval
From the repository root, with dependencies installed:

```bash
python -m apps.finetuning.dense
python -m apps.finetuning.dense_crossencoder
python -m apps.indexing.dense
```

Fine-tuning creates directories for the fine-tuned Qwen (bi-encoder) and BGE (cross-encoder) models. Indexing builds the paragraph and paper indexes with both the pre-trained and fine-tuned Qwen models. See further details in `apps/finetuning/README.md` and `apps/indexing/README.md`.

##### Evaluation
```bash
python -m apps.evaluation.evaluate
```

Performs retrieval and evaluation based on the settings in `configs/eval.yaml`. See `apps/evaluation/README.md` for more details.

## Team Member Contributions

### Clayton Durepos

#### Project Code

- Sparse Neural Model Experiment Code: 
`finetuning/sparse.py`, `indexing/sparse.py`, `retrieval/sparse.py` 
- Evaluation Pipeline
`evaluation/evaluate.py`

#### Project Report
- Introduction
- Sparse Model Methodology
- Evaluation Methodology
- Results

###  Ella Hawkins

#### Project Code

- Dense Neural Model Experiment Code:
`finetuning/dense.py`, `finetuning/dense_crossencoder.py`, `indexing/dense.py`, `retrieval/dense.py`
- Bash Scripts
`install.sh`, `run.sh`

#### Project Report
- Abstract
- Dense Model Methodology
- Analysis
- Conclusion