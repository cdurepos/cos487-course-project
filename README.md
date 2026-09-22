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
data/                  Project data
indexes/               Retrieval indexes
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
Users must download course data files. For installation, ~1.9GB of disk space is required. Arrange data files as specified below.
```text
data/
├── JSON Files/     Unzipped data corpus
├── Study.json      Query file
└── processed/      Written by the preprocessing script
```

