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
Each part of the project lives in its own folder under `apps/`, so the Python
work and the interface work stay out of each other's way.

```text
apps/
├── processing/        Prepares the corpus: cleaning, stemming, token files
├── backend/           Search service that will rank queries — empty for now
└── frontend/          Search interface (React + Vite)
    └── src/
        ├── api/         Where results come from — mock data until the service exists
        ├── components/  The screen: search bar, results, filters, side panel, dialogs
        └── hooks/       Shared browser-side state, such as recent searches
```

`processing` produces the token files, `backend` ranks against them and answers a
query over HTTP, and `frontend` renders whatever comes back. The corpus itself and
everything built from it stay out of the repository — each of us downloads the data
and rebuilds locally.

