# Evaluation

Compares BM25 and TF-IDF on the Study qrels using [ranx](https://github.com/AmenRa/ranx), and writes TREC run files you can inspect later.

## Run

From the repository root:

```bash
pip install -r requirements.txt
python -m apps.evaluation.evaluate
```

All settings are in `config.yaml`. Edit that file, then re-run.

## What you get

| Output | Location |
|--------|----------|
| TREC runs | `data/runs/{method}/` — e.g. `bm25/paper_stem.tsv`, `tfidf/paragraph_no-stem.tsv` |
| Metric summary | `data/evaluation/` — e.g. `paper_stem.json`, `paragraph_no-stem.json` |

The console also prints per-method scores and a BM25 vs TF-IDF comparison table.

## Quick smoke test

In `config.yaml`, set `limit: 10`, run once, then set `limit` back to `null` for the full Study set.
