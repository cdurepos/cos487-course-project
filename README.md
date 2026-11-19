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
configs/               Evaluation settings
bin/                   Install and run scripts
apps/
├── dense/             Qwen fine-tuning, indexing, and search
├── evaluation/        Run retrieval and score the results
└── utils.py           Paths shared by more than one module
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

Edit `configs/eval.yaml` to change query sets, metrics, levels, and which Qwen runs to score. See `apps/evaluation/README.md` for more details.

To set up environment and install dependencies:

```bash
bash bin/install.sh
```

To run retrieval and evaluation:

```bash
bash bin/run.sh --eval
```

`bash bin/run.sh --prod` starts the search interface.

### Manual Run
Install Python dependencies once from the repository root:

```bash
pip install -r requirements.txt
```

##### Dense Retrieval
From the repository root, with dependencies installed:

```bash
python -m apps.dense.finetune
python -m apps.dense.indexing
```

Fine-tuning writes the Qwen adapter under `models/finetuned_qwen`. Indexing builds the paragraph and paper indexes for the pre-trained and fine-tuned models. See `apps/dense/README.md`.

##### Evaluation
```bash
python -m apps.evaluation.evaluate
```

Performs retrieval and evaluation based on the settings in `configs/eval.yaml`. See `apps/evaluation/README.md` for more details.

## Course Required Documentation

### Team Member Contributions

#### Clayton Durepos

- Part 2 sparse experiments (removed for Part 3)
- Evaluation pipeline: `evaluation/evaluate.py`

#### Ella Hawkins

- Dense model code: `dense/finetune.py`, `dense/indexing.py`, `dense/retrieval.py`
- Indexer component and BM25 system
- `install.sh`, `run.sh`
