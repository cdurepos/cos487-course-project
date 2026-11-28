"""
Grid search for BM25 k1 and b parameters.

Evaluates all combinations of k1 and b on the original Train questions
and on the generated Train replies, for each prompt approach, at the
stemmed paper and paragraph indexes.

Results are evaluated using nDCG, nDCG@5, MRR, Precision@5,
and Precision@10, and saved as CSV files under data/metrics/grid-search/.

Run from the repository root:
    python apps/grid-search.py
    python apps/grid-search.py --approach original,rewrite
"""


from __future__ import annotations

import argparse
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

from apps.retrieval import load_config, retrieve
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


def load_original_queries() -> list[tuple[str, str, str]]:
    """Load the Train questions with no generated rewrite.

    Each item is (query_id, question, question). Retrieval searches that
    text as one query.
    """
    rows = json.loads((QUERY_DIR / "Train.json").read_text())
    return [
        (str(row["query_id"]), row["query"], row["query"])
        for row in rows
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
                approach=None if approach == "original" else approach,
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
        / f"{approach}_{level}_{stem_name}.csv"
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

    The file is keyed by approach name, then paper or paragraph. Retrieval
    reads that block when no k1 or b is passed. Other approaches are left in
    place. The winning metric values are stored beside the parameters.
    """
    current = load_config()
    section = current.setdefault(approach, {})
    if not isinstance(section, dict):
        section = {}
        current[approach] = section
    section[level] = {
        "k_1": best["k1"],
        "b": best["b"],
        **{metric: best[metric] for metric in METRICS},
    }
    CONFIG_PATH.write_text(
        "# BM25 parameters for the stemmed indexes.\n"
        + yaml.safe_dump(current, sort_keys=False)
    )


def selected_systems(values: list[str] | None) -> list[str]:
    """Return the systems named by --approach.

    Each value may be one name or a comma-separated list. An omitted flag
    selects the original queries and every generated approach.
    """
    known = ("original", *PROMPT_MAP)
    if not values:
        return list(known)
    chosen = []
    unknown = []
    for value in values:
        for name in value.split(","):
            name = name.strip()
            if not name or name in chosen:
                continue
            if name not in known:
                unknown.append(name)
                continue
            chosen.append(name)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown approach: {', '.join(unknown)}. "
            f"Choose from: {', '.join(known)}"
        )
    if not chosen:
        raise argparse.ArgumentTypeError(
            f"provide at least one approach. Choose from: {', '.join(known)}"
        )
    return chosen


def parse_args() -> list[str]:
    """Read --approach and return the systems to search."""
    known = ("original", *PROMPT_MAP)
    parser = argparse.ArgumentParser(
        description="Grid search BM25 k1 and b on selected systems.",
    )
    parser.add_argument(
        "--approach",
        action="append",
        metavar="NAME",
        help=(
            "System to search. Repeat the flag or pass a comma-separated list. "
            f"Default: {', '.join(known)}"
        ),
    )
    args = parser.parse_args()
    try:
        return selected_systems(args.approach)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))


def main():
    """Runs BM25 grid search."""

    systems = parse_args()
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
    total_evaluations = len(systems) * len(LEVELS) * total_combinations
    print(
        f"Total evaluations: "
        f"{len(systems)} systems × "
        f"{len(LEVELS)} levels × "
        f"{total_combinations} combinations = "
        f"{total_evaluations}"
    )
    overall_start = time.perf_counter()
    for approach in systems:
        if approach == "original":
            queries = load_original_queries()
            print(
                f"\nLoaded {len(queries)} original Train queries.",
                flush=True,
            )
        else:
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
