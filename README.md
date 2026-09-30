# COS487 Information Retrieval - Course Project
This is a repository intended to host the soon-to-be developed code for the course project assigned in COS487 Information Retrieval at the University of Southern Maine.

## Team Details
Name: Search Party

Members:
- Clayton Durepos (Contact)
- Ella Hawkins
- Jack Bergin
- Grace Kalonji

## Repository Structure
Each part of the project has its own folder under `apps/`.

```text
data/                  Project data (corpus, processed tokens, indexes)
bin/                   Install and run scripts
apps/
├── processing/        Scripts for pre-processing data
├── retrieval/         Retrieval systems and index script
├── evaluation/        Evaluation scripts
├── prod/
    ├── backend/           Search service backend
    └── frontend/          Search interface
        └── src/
            ├── api/         Files for API-related functions
            ├── components/  UI components
            └── hooks/       Shared browser-side state, such as recent searches
```

## Use Guide

### Data Setup
Users must download COS487 Information Retrieval course data files. For installation, ~9GB of disk space is required. Arrange data files as specified below.
```text
data/
├── JSON Files/     Unzipped data corpus
├── qrels/          QREL files
└── Study.json      Query file
```

### Quick Run
To launch this application on Linux or MacOS, first set up the data (as outlined above). Then, run the following commands in a bash terminal from the repository root. By default, the backend runs on `localhost:8000` and the frontend runs on `localhost:5173`.

To set up environment and install dependencies:

```bash
bash bin/install.sh
```

`bin/run.sh` takes the mode to run as its first argument. To launch the search interface:

```bash
bash bin/run.sh --prod
```

To run the research pipeline:

```bash
bash bin/run.sh --eval
```

Edit `apps/evaluation/config.yaml` to change metrics, levels, stemming.
See `apps/evaluation/README.md` for more details.

### Manual Run
Install Python dependencies once from the repository root:

```bash
pip install -r requirements.txt
```

##### Environment Build
```bash
python -m apps.processing.preprocess
python -m apps.retrieval.index
```

##### Backend
From the repository root:

```bash
uvicorn apps.prod.backend.main:app --reload --port 8000
```

##### Frontend
In a second terminal:

```bash
cd apps/prod/frontend
npm install
npm run dev
```

##### Evaluation
```bash
python -m apps.evaluation.evaluate
```


