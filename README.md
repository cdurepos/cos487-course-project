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

## Data Setup
Users must download COS487 Information Retrieval course data files. For installation, ~1.9GB of disk space is required. Arrange data files as specified below.
```text
data/
├── JSON Files/     Unzipped data corpus
├── qrels/          QREL files
└── Study.json      Query file
```
## Production App
Install Python dependencies once from the repository root:

```bash
pip install -r requirements.txt
```

### Backend
From the repository root:

```bash
uvicorn apps.prod.backend.main:app --reload --port 8000
```

### Frontend
In a second terminal:

```bash
cd apps/prod/frontend
npm install
npm run dev
```

## Evaluation (BM25 vs TF-IDF)
Uses the Study qrels in `data/qrels/` and the [ranx](https://github.com/AmenRa/ranx) library.
Settings live in `apps/evaluation/config.yaml`.

```bash
pip install -r requirements.txt
python -m apps.evaluation.evaluate
```

Edit the YAML to change metrics, levels, stemming, or set `limit: 10` for a smoke test, then re-run.
See `apps/evaluation/README.md` for details.
