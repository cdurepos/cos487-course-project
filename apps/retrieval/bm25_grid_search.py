"""
Grid search for BM25 k1 and b parameters.

Evaluates all combinations of k1 and b across:
    - paper- and paragraph-level retrieval
    - stemmed and unstemmed indexes

Results are evaluated using nDCG, nDCG@5, MRR, Precision@5,
and Precision@10, and saved as CSV files under data/evaluation/.

Run from the repository root:
    python -m apps.retrieval.bm25_grid_search
"""


from __future__ import annotations

import csv
import time
from pathlib import Path
from tqdm import tqdm

import numpy as np
from ranx import Qrels, Run, evaluate

from apps.retrieval import bm25
from apps.evaluation.evaluate import (
    DATA_DIR,
    METRICS_DIR,
    load_queries,
    load_qrels,
)


ROOT = Path(__file__).resolve().parents[2]

LEVELS = [
    # "paper",
    "paragraph"
]

STEM_OPTIONS = [
    True,
    False
]

# Number of documents retained for each query/run.
K = 100

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
    3.0,
]

B_VALUES = [
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
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


def prepare_all_queries(
    queries: list[tuple[str, str]],
    level: str,
    stem: bool,
) -> list[tuple[str, dict]]:
    """
    Prepare all queries once for a level/stemming combination.

    Args:
        queries (list[tuple[str, str]]): List of (query_id, query_text) tuples.
        level (str): The retrieval level ("paper" or "paragraph").
        stem (bool): Whether to use the stemmed index.
    
    Returns:
        list[tuple[str, dict]]: List of (query_id, prepared_query) tuples
    """

    print(
        f"Preparing {len(queries)} queries "
        f"for level={level}, stem={stem}...",
        flush=True,
    )
    started = time.perf_counter()
    prepared = bm25.prepare_queries(
        queries,
        level,
        stem,
    )
    elapsed = time.perf_counter() - started
    print(
        f"Prepared {len(prepared)} queries "
        f"in {elapsed:.2f}s.",
        flush=True,
    )
    return prepared


def retrieve_all_parameters(
    prepared_query: dict,
    parameter_grid: list[tuple[float, float]],
    k: int,
) -> list[dict[str, float]]:
    """
    Evaluates every k1/b combination for one prepared query.

    Args:
        prepared_query (dict): The prepared query representation.
        parameter_grid (list[tuple[float, float]]): List of (k1, b) parameter combinations to evaluate.
        k (int): The number of top documents to return for each parameter combination.

    Returns:
        list[dict[str, float]]: A list of dictionaries, one for each parameter combination, 
                                where each dictionary maps document IDs to their BM25 scores for that combination.
    """

    postings = prepared_query["postings"]
    average_doc_length = prepared_query["average_doc_length"]
    num_parameters = len(parameter_grid)

    if not postings or average_doc_length <= 0:
        return [{} for _ in range(num_parameters)]

    k1_values = np.asarray(
        [k1 for k1, _ in parameter_grid],
        dtype=np.float64,
    )
    b_values = np.asarray(
        [b for _, b in parameter_grid],
        dtype=np.float64,
    )
    scores = [
        {}
        for _ in range(num_parameters)
    ]
    for posting in postings:
        doc_ids = posting["doc_ids"]
        tf = posting["tf"]
        doc_lengths = posting["doc_lengths"]
        idf = posting["idf"]
        
        tf_column = tf[:, None]
        doc_length_column = doc_lengths[:, None]

        denominator = (
            tf_column
            + k1_values[None, :]
            * (
                1.0
                - b_values[None, :]
                + b_values[None, :]
                * doc_length_column
                / average_doc_length
            )
        )
        term_scores = (
            idf
            * (
                tf_column
                * (k1_values[None, :] + 1.0)
            )
            / denominator
        )
        for parameter_index in range(num_parameters):
            parameter_scores = term_scores[:, parameter_index]
            score_dict = scores[parameter_index]
            for doc_id, score in zip(
                doc_ids,
                parameter_scores,
            ):
                score_dict[doc_id] = (
                    score_dict.get(doc_id, 0.0)
                    + float(score)
                )
    results = []
    for score_dict in scores:
        if not score_dict:
            results.append({})
            continue
        top_results = sorted(
            score_dict.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:k]
        results.append(dict(top_results))
    return results


def build_runs(
    prepared_queries: list[tuple[str, dict]],
    parameter_grid: list[tuple[float, float]],
    level: str,
    stem: bool,
) -> list[Run]:
    """
    Build a ranx Run for every k1/b combination.
    
    Args:
        prepared_queries (list[tuple[str, dict]]): List of (query_id, prepared_query) tuples.
        parameter_grid (list[tuple[float, float]]): List of (k1, b) parameter combinations to evaluate.
        level (str): The retrieval level ("paper" or "paragraph").
        stem (bool): Whether to use the stemmed index.
    
    Returns:
        list[Run]: A list of ranx Run objects, one for each parameter combination.
    """

    num_parameters = len(parameter_grid)
    ranked_runs: list[dict[str, dict[str, float]]] = [
        {}
        for _ in range(num_parameters)
    ]
    query_number = 1
    for qid, prepared_query in tqdm(prepared_queries, desc="Retrieving queries", unit="query"):
        query_results = retrieve_all_parameters(
            prepared_query,
            parameter_grid,
            K,
        )
        for parameter_index, results in enumerate(query_results):
            ranked_runs[parameter_index][qid] = results
        query_number += 1
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
    queries: list[tuple[str, str]],
    level: str,
    stem: bool,
) -> list[dict]:
    """
    Evaluate the complete BM25 parameter grid for one level/stemming combination.

    Args:
        queries (list[tuple[str, str]]): List of (query_id, query_text) tuples.
        level (str): The retrieval level ("paper" or "paragraph").
        stem (bool): Whether to use the stemmed index.
    
    Returns:
        list[dict]: A list of dictionaries, one for each parameter combination,
                    containing the evaluation metrics for that combination.
    """

    print(
        f"\n{'=' * 75}\n"
        f"BM25 GRID SEARCH\n"
        f"level={level}, stem={stem}, k={K}\n"
        f"{'=' * 75}",
        flush=True,
    )
    query_ids = {
        query[0]
        for query in queries
    }
    qrels = load_qrels(
        "train",
        level,
        query_ids,
        1,
    )
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
    prepared_queries = prepare_all_queries(
        queries,
        level,
        stem,
    )
    started = time.perf_counter()
    runs = build_runs(
        prepared_queries,
        parameter_grid,
        level,
        stem,
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
) -> Path:
    """
    Save grid-search results to CSV.

    Args:
        results (list[dict]): List of dictionaries containing evaluation metrics for each parameter combination.
        level (str): The level of the queries.
        stem (bool): Whether the queries are stemmed.

    Returns:
        Path: The path to the saved CSV file.
    """

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    stem_name = (
        "stemmed"
        if stem
        else "unstemmed"
    )
    path = (
        METRICS_DIR
        / (
            f"bm25_grid_"
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


def main():
    """Runs BM25 grid search."""

    queries = load_queries("train")
    print(
        f"Loaded {len(queries)} queries."
    )
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
    total_evaluations = (
        len(LEVELS)
        * len(STEM_OPTIONS)
        * total_combinations
    )
    print(
        f"Total evaluations: "
        f"{len(LEVELS)} levels × "
        f"{len(STEM_OPTIONS)} stemming settings × "
        f"{total_combinations} combinations = "
        f"{total_evaluations}"
    )
    overall_start = time.perf_counter()
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
    elapsed = time.perf_counter() - overall_start
    print(
        f"\nGrid search complete "
        f"in {elapsed:.2f}s."
    )


if __name__ == "__main__":
    main()
