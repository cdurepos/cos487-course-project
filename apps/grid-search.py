"""
Grid search for BM25 k1 and b parameters.

Evaluates all combinations of k1 and b on the generated Train replies,
for each prompt approach, at the stemmed paper and paragraph indexes.

Results are evaluated using nDCG, nDCG@5, MRR, Precision@5,
and Precision@10, and saved as CSV files under data/metrics/grid-search/.

Run from the repository root:
    python apps/grid-search.py
"""


from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from tqdm import tqdm
import yaml

from ranx import Qrels, Run, evaluate

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from apps.retrieval import retrieve
from apps.utils import CONFIG_PATH, GENERATED_DIR, PROMPT_MAP, QRELS_DIR, QUERY_DIR, ROOT_DIR, TRAINING_DIR

LEVELS = [
    "paper",
    "paragraph"
]

# Number of documents retained for each query/run.
K = 100

K1_VALUES = [
    0.9,
    1.2,    # default
    1.6
]

B_VALUES = [
    0.4,
    0.75,   # default
    1.0
]

METRICS = [
    "ndcg",
    "ndcg@5",
    "mrr",
    "precision@5",
    "precision@10",
]


def make_parameter_grid() -> list[tuple[float, float]]:
    """Return all k1/b combinations in deterministic order."""
    return [
        (k1, b)
        for k1 in K1_VALUES
        for b in B_VALUES
    ]


def load_train_queries(approach: str) -> list[tuple[str, str, str]]:
    """Load generated Train replies for one approach.

    Each item is (query_id, generated reply, original question). Retrieval
    parses the reply and falls back to the original question when it is empty.
    """
    generated_path = GENERATED_DIR / f"train_{approach}.json"
    if not generated_path.is_file():
        raise FileNotFoundError(generated_path)
    replies = json.loads(generated_path.read_text())["queries"]
    rows = json.loads((QUERY_DIR / "Train.json").read_text())
    return [
        (str(row["query_id"]), replies.get(str(row["query_id"]), ""), row["query"])
        for row in rows
    ]


def build_runs(
    queries: list[tuple[str, str, str]],
    parameter_grid: list[tuple[float, float]],
    level: str,
    stem: bool,
    approach: str,
) -> list[Run]:
    """
    Build a ranx Run for every k1/b combination.

    Each search goes through retrieval.retrieve, which parses multi-line and
    weighted replies and fuses those rankings.

    Args:
        queries: (query_id, generated reply, original question) triples.
        parameter_grid (list[tuple[float, float]]): List of (k1, b) parameter combinations to evaluate.
        level (str): The retrieval level ("paper" or "paragraph").
        stem (bool): Whether to use the stemmed index.
        approach (str): Generated prompt approach passed to retrieval.

    Returns:
        list[Run]: A list of ranx Run objects, one for each parameter combination.
    """

    ranked_runs: list[dict[str, dict[str, float]]] = []
    for k1, b in parameter_grid:
        ranked = {}
        for qid, reply, original in tqdm(
            queries,
            desc=f"k1={k1:g} b={b:g}",
            unit="query",
        ):
            ranking = retrieve(
                reply,
                approach=approach,
                level=level,
                k=K,
                original=original,
                k_1=k1,
                b=b,
            )
            ranked[qid] = dict(ranking)
        ranked_runs.append(ranked)
    runs = []
    for parameter_index, (k1, b) in enumerate(parameter_grid):
        name = (
            f"bm25_{level}_"
            f"{'stem' if stem else 'no-stem'}_"
            f"k1-{k1:g}_b-{b:g}"
        )
        runs.append(
            Run.from_dict(
                ranked_runs[parameter_index],
                name=name,
            )
        )
    return runs


def evaluate_level(
    queries: list[tuple[str, str, str]],
    level: str,
    stem: bool,
    approach: str,
) -> list[dict]:
    """
    Evaluate the complete BM25 parameter grid for one approach and level.

    Args:
        queries: (query_id, generated reply, original question) for each Train query.
        level (str): The retrieval level ("paper" or "paragraph").
        stem (bool): Whether to use the stemmed index.
        approach (str): Generated prompt approach, such as "rewrite" or "multi".
    
    Returns:
        list[dict]: A list of dictionaries, one for each parameter combination,
                    containing the evaluation metrics for that combination.
    """

    print(
        f"\n{'=' * 75}\n"
        f"BM25 GRID SEARCH\n"
        f"approach={approach}, level={level}, stem={stem}, k={K}\n"
        f"{'=' * 75}",
        flush=True,
    )
    query_ids = {
        query[0]
        for query in queries
    }
    qrel_path = QRELS_DIR / f"Train_{level}_qrel.tsv"
    if not qrel_path.exists():
        raise FileNotFoundError(qrel_path)
    judged = Qrels.from_file(str(qrel_path), kind="trec").to_dict()
    qrels = Qrels.from_dict({qid: docs for qid, docs in judged.items() if qid in query_ids})
    qrels.set_relevance_level(rel_lvl=1)
    qrels_dict = qrels.to_dict()
    qrels = Qrels.from_dict(
        {
            qid: docs
            for qid, docs in qrels_dict.items()
            if qid in query_ids
        }
    )
    parameter_grid = make_parameter_grid()
    print(
        f"Queries: {len(queries)}"
        f" | Combinations: {len(parameter_grid)}"
        f" | k1 values: {len(K1_VALUES)}"
        f" | b values: {len(B_VALUES)}",
        flush=True,
    )
    started = time.perf_counter()
    runs = build_runs(
        queries,
        parameter_grid,
        level,
        stem,
        approach,
    )
    retrieval_elapsed = time.perf_counter() - started
    print(
        f"\nRetrieval completed in "
        f"{retrieval_elapsed:.2f}s.",
        flush=True,
    )
    results = []
    for (k1, b), run in zip(parameter_grid, runs):
        scores = evaluate(
            qrels,
            run,
            METRICS,
            make_comparable=True,
        )
        results.append(
            {
                "k1": k1,
                "b": b,
                **{
                    metric: float(scores[metric])
                    for metric in METRICS
                },
            }
        )
    print(
        f"Evaluation complete for "
        f"{len(parameter_grid)} combinations.",
        flush=True,
    )
    return results


def print_results(
    results: list[dict],
    sort_metric: str = "ndcg",
) -> None:
    """
    Print all results sorted by the selected metric.
    
    Args:
        results (list[dict]): List of dictionaries containing evaluation metrics for each parameter combination.
        sort_metric (str): The metric to sort the results by. Default is "ndcg".
    """

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
    approach: str,
) -> Path:
    """
    Save grid-search results to CSV.

    Args:
        results (list[dict]): List of dictionaries containing evaluation metrics for each parameter combination.
        level (str): The level of the queries.
        stem (bool): Whether the queries are stemmed.
        approach (str): Generated prompt approach these results were scored on.

    Returns:
        Path: The path to the saved CSV file.
    """

    TRAINING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    stem_name = (
        "stemmed"
        if stem
        else "unstemmed"
    )
    path = (
        TRAINING_DIR
        / (
            f"bm25_grid_{approach}_"
            f"{level}_{stem_name}.csv"
        )
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


def save_config(level: str, approach: str, best: dict) -> None:
    """Write the best k1 and b for one approach and level into config.yaml.

    Retrieval reads that block when no k1 or b is passed. The winning metric
    values are stored beside the parameters.
    """
    current = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    section = current.setdefault(level, {})
    section[approach] = {
        "k_1": best["k1"],
        "b": best["b"],
        **{metric: best[metric] for metric in METRICS},
    }
    CONFIG_PATH.write_text(
        "# BM25 parameters for the stemmed indexes.\n"
        + yaml.safe_dump(current, sort_keys=False)
    )


def main():
    """Runs BM25 grid search."""

    total_combinations = (
        len(K1_VALUES)
        * len(B_VALUES)
    )
    print(
        f"\nGrid: "
        f"{len(K1_VALUES)} k1 values × "
        f"{len(B_VALUES)} b values = "
        f"{total_combinations} combinations"
    )
    total_evaluations = len(PROMPT_MAP) * len(LEVELS) * total_combinations
    print(
        f"Total evaluations: "
        f"{len(PROMPT_MAP)} approaches × "
        f"{len(LEVELS)} levels × "
        f"{total_combinations} combinations = "
        f"{total_evaluations}"
    )
    overall_start = time.perf_counter()
    for approach in PROMPT_MAP:
        queries = load_train_queries(approach)
        print(
            f"\nLoaded {len(queries)} generated Train queries "
            f"for {approach}.",
            flush=True,
        )
        for level in LEVELS:
            results = evaluate_level(
                queries=queries,
                level=level,
                stem=True,
                approach=approach,
            )
            print_results(
                results,
                sort_metric="ndcg",
            )
            path = save_results(
                results=results,
                level=level,
                stem=True,
                approach=approach,
            )
            best = max(results, key=lambda row: (row["ndcg"], row["mrr"]))
            save_config(level, approach, best)
            print(
                f"\nSaved results → "
                f"{path.relative_to(ROOT_DIR)}",
                flush=True,
            )
            print(
                f"Saved {approach}/{level} parameters → "
                f"k1={best['k1']:g}, b={best['b']:g}, nDCG={best['ndcg']:.4f}",
                flush=True,
            )
    elapsed = time.perf_counter() - overall_start
    print(
        f"\nGrid search complete "
        f"in {elapsed:.2f}s."
    )


if __name__ == "__main__":
    main()
