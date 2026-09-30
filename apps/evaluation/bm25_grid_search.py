"""
Grid search for BM25 k1 and b parameters using the course qrels and ranx.

Runs all combinations of:
    level:  paper, paragraph
    stem:   True, False

Run from the repository root:
    python -m apps.evaluation.bm25_grid_search

Results are printed to the terminal and saved as CSV files under:
    data/evaluation/
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from ranx import Qrels, Run, evaluate

from apps.retrieval import bm25

from apps.evaluation.evaluate import (
    DATA_DIR,
    METRICS_DIR,
    load_queries,
    load_qrels,
)


ROOT = Path(__file__).resolve().parents[2]

SET_NAME = "study"

LEVELS = [
    "paper",
    "paragraph",
]

STEM_OPTIONS = [
    True,
    False,
]

# Documents retrieved for each query.
K = 100

# Parameter grid.
K1_VALUES = [
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
    2.25,
    2.5,
    2.75,
    3.0
]

B_VALUES = [
    0.0,
    0.25,
    0.5,
    0.75,
    1.0
]

# Metrics used for evaluation.
METRICS = [
    "ndcg",
    "ndcg@5",
    "mrr",
    "precision@5",
    "precision@10",
]


def build_run(
    queries: list[tuple[str, str]],
    level: str,
    stem: bool,
    k1: float,
    b: float,
) -> Run:
    """Run BM25 for every query using one k1/b combination."""
    name = (
        f"bm25_{level}_"
        f"{'stem' if stem else 'no-stem'}_"
        f"k1-{k1:g}_b-{b:g}"
    )
    ranked: dict[str, dict[str, float]] = {}
    for qid, text in queries:
        hits = bm25.retrieve(
            text,
            level,
            stem,
            K,
            k_1=k1,
            b=b,
        )
        ranked[qid] = {
            doc_id: float(score)
            for doc_id, score in hits.items()
        }
    return Run.from_dict(ranked, name=name)


def evaluate_parameters(
    qrels: Qrels,
    queries: list[tuple[str, str]],
    level: str,
    stem: bool,
    k1: float,
    b: float,
) -> dict[str, float]:
    """Evaluate one BM25 parameter combination."""
    run = build_run(
        queries=queries,
        level=level,
        stem=stem,
        k1=k1,
        b=b,
    )
    scores = evaluate(
        qrels,
        run,
        METRICS,
        make_comparable=True,
    )
    return {
        metric: float(scores[metric])
        for metric in METRICS
    }


def evaluate_level(
    queries: list[tuple[str, str]],
    level: str,
    stem: bool,
) -> list[dict]:
    """Run the complete grid search for one level/stemming combination."""
    print(
        f"\n{'=' * 75}\n"
        f"BM25 GRID SEARCH\n"
        f"set={SET_NAME}, level={level}, stem={stem}, k={K}\n"
        f"{'=' * 75}",
        flush=True,
    )
    qrels = load_qrels(SET_NAME, level)
    query_ids = {qid for qid, _ in queries}
    qrels_all = qrels.to_dict()
    qrels = Qrels.from_dict(
        {
            qid: docs
            for qid, docs in qrels_all.items()
            if qid in query_ids
        }
    )
    combinations = [
        (k1, b)
        for k1 in K1_VALUES
        for b in B_VALUES
    ]
    print(
        f"Queries: {len(queries)}"
        f" | Combinations: {len(combinations)}"
        f" | k1 values: {len(K1_VALUES)}"
        f" | b values: {len(B_VALUES)}",
        flush=True,
    )
    results: list[dict] = []
    started = time.perf_counter()
    for i, (k1, b) in enumerate(combinations, start=1):
        combo_start = time.perf_counter()
        scores = evaluate_parameters(
            qrels=qrels,
            queries=queries,
            level=level,
            stem=stem,
            k1=k1,
            b=b,
        )
        result = {
            "k1": k1,
            "b": b,
            **scores,
        }
        results.append(result)
        elapsed = time.perf_counter() - combo_start
        print(
            f"  [{i:2d}/{len(combinations)}] "
            f"k1={k1:4.2f}, b={b:4.2f} | "
            f"nDCG={scores['ndcg']:.4f} | "
            f"nDCG@5={scores['ndcg@5']:.4f} | "
            f"MRR={scores['mrr']:.4f} | "
            f"P@5={scores['precision@5']:.4f} | "
            f"P@10={scores['precision@10']:.4f} | "
            f"{elapsed:.1f}s",
            flush=True,
        )
    total_elapsed = time.perf_counter() - started
    print(
        f"\nCompleted {len(combinations)} combinations "
        f"in {total_elapsed:.1f}s."
    )
    return results


def print_results(
    results: list[dict],
    sort_metric: str = "ndcg",
) -> None:
    """Print all results sorted by the selected metric."""
    results = sorted(
        results,
        key=lambda row: row[sort_metric],
        reverse=True,
    )
    print(
        f"\n--- Results sorted by {sort_metric} ---"
    )
    header = (
        f"{'Rank':>4} "
        f"{'k1':>6} "
        f"{'b':>6} "
        f"{'nDCG':>10} "
        f"{'nDCG@5':>10} "
        f"{'MRR':>10} "
        f"{'P@5':>10} "
        f"{'P@10':>10}"
    )
    print(header)
    print("-" * len(header))
    for rank, row in enumerate(results, start=1):
        print(
            f"{rank:4d} "
            f"{row['k1']:6.2f} "
            f"{row['b']:6.2f} "
            f"{row['ndcg']:10.4f} "
            f"{row['ndcg@5']:10.4f} "
            f"{row['mrr']:10.4f} "
            f"{row['precision@5']:10.4f} "
            f"{row['precision@10']:10.4f}"
        )


def save_results(
    results: list[dict],
    level: str,
    stem: bool,
) -> Path:
    """Save grid-search results to CSV."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    stem_name = "stemmed" if stem else "unstemmed"
    path = (
        METRICS_DIR
        / f"bm25_grid_{SET_NAME}_{level}_{stem_name}.csv"
    )
    fieldnames = [
        "k1",
        "b",
        *METRICS,
    ]
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(results)
    return path


def main() -> int:
    """Run BM25 grid search for every level and stemming setting."""
    query_path = DATA_DIR / "Study.json"
    queries = load_queries(SET_NAME)
    print(
        f"Loaded {len(queries)} queries "
        f"from {query_path.name}"
    )
    print(
        f"\nGrid: "
        f"{len(K1_VALUES)} k1 values × "
        f"{len(B_VALUES)} b values = "
        f"{len(K1_VALUES) * len(B_VALUES)} combinations"
    )
    print(
        f"Total evaluations: "
        f"{len(LEVELS)} levels × "
        f"{len(STEM_OPTIONS)} stemming settings × "
        f"{len(K1_VALUES) * len(B_VALUES)} combinations = "
        f"{len(LEVELS) * len(STEM_OPTIONS) * len(K1_VALUES) * len(B_VALUES)}"
    )
    for level in LEVELS:
        for stem in STEM_OPTIONS:
            results = evaluate_level(
                queries=queries,
                level=level,
                stem=stem,
            )
            print_results(
                results,
                sort_metric="ndcg",
            )
            path = save_results(
                results=results,
                level=level,
                stem=stem,
            )
            print(
                f"\nSaved results → "
                f"{path.relative_to(ROOT)}",
                flush=True,
            )
    print("\nGrid search complete.")


if __name__ == "__main__":
    main()
