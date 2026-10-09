# Evaluation

Runs the retrieval systems on every query set in `config.yaml`. Each set writes TREC run files. Sets with `metrics: true` are measured against qrels using [ranx](https://github.com/AmenRa/ranx) and record systems performance. Sets with `metrics: false` write runs only.

## Run

From the repository root:

```bash
pip install -r requirements.txt
python -m apps.evaluation.evaluate
```

All settings, including which sets get metrics and the team name used in output filenames, are in `config.yaml`. Edit that file, then re-run.

The committed config retrieves Study and Test. Study has `metrics: true`, including systems performance. Test writes runs only, for annotation or any other use of the ranked results. A missing qrel file is an error when `metrics` is true.

## Metrics

Each level is scored twice, because the qrels use graded judgments
(2 = Relevant, 1 = Partially Relevant):

| Pass | Relevance level | Reports |
|------|-----------------|---------|
| Relevant + Partially Relevant | `1` | nDCG, nDCG@5, MRR, P@5, P@10 |
| Relevant only | `2` | MRR, P@5 |

## Significance Testing
Each pass also runs a paired significance test between BM25 and TF/IDF.
`better` is `null` when the gap is not significant. Win/tie/loss counts queries from the first listed method's perspective:

```json
"bm25 vs tfidf": {
  "ndcg": { "p": 0.00323, "better": "bm25", "win_tie_loss": { "W": 6, "T": 5, "L": 1 } },
  "mrr":  { "p": 0.0987,  "better": null,   "win_tie_loss": { "W": 4, "T": 6, "L": 2 } }
}
```

The same two passes are also saved per query, one row per query id, in `<set>_<level>_queries.json`. The summary keeps the means.

## Systems performance

Each summary includes a `performance` object:

| Field | Meaning |
|-------|---------|
| `index.bytes` | Size of the index file used for that level |
| `latency_ms.<method>.mean` | Mean steady-state query time, in milliseconds |
| `latency_ms.<method>.median` | Median steady-state query time, in milliseconds |

Systems performance is recorded only for sets with `metrics: true`. In the committed config, that is Study. The first query for each method and index is left out of the mean and median. That call loads the index, and for TF-IDF it also builds document norms. `queries` is the number of timed queries; `warmup_excluded` is how many were left out.

## What you get

| Output | Location |
|--------|----------|
| TREC runs | `data/runs/<team>_<level>_<method>_<set>.tsv` |
| Metric summary | `data/evaluation/<set>_<level>.json` |
| Per-query metrics | `data/evaluation/<set>_<level>_queries.json` |
| Significance report | `data/evaluation/<set>_<level>_significance.json` |

Runs use the standard 6-column TREC format: `qid Q0 docid rank score run_tag`.

Stemming is the default. Setting `stem: false` writes `_no-stem` files.

## Quick smoke test

In `config.yaml`, set `limit: 10`, run once, then set `limit` back to `null`. The limit applies to every set.
