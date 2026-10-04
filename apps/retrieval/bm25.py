"""
Contains the BM25 retrieval system

Caches indexes, so that retrieval can be done from multiple indexes without
having to reload them every time.

Uses BM25 parameters from config/bm25.yaml by default.

To create a REPL command-line interface, run this file directly from the root:

    python -m apps.retrieval.bm25_system
"""

import math
from pathlib import Path
import yaml
import numpy as np

from apps.processing.preprocess import preprocess
import apps.retrieval.index as indexer


CONFIG_PATH = Path(__file__).resolve().parent / "bm25_config.yaml"


def load_config() -> dict:
    """Load the BM25 configuration from YAML."""

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}
    except FileNotFoundError:
        raise FileNotFoundError(
            f"BM25 config file not found: {CONFIG_PATH}"
        )


CONFIG = load_config()

indexes = {}


index_id = lambda level, stem: f"{level}_{'un' if not stem else ''}stemmed"

def load_index(level: str, stem: bool) -> dict:
    """Loads an inverted index for the given level

    Args:
        level (str): The level of the index to load. Can be either "paragraph" or "paper"
        stem (bool): Whether to use stemming or not

    Returns:
        dict: The inverted index
    """

    global indexes
    id = index_id(level, stem)
    if id not in indexes:
        print(f"Loading index for level '{level}' with {'stemming' if stem else 'no stemming'}...")
        indexes[id] = indexer.load_index(level, stem)
    return indexes[id]


def get_bm25_params(
    level: str,
    stem: bool,
    k_1: float | None = None,
    b: float | None = None,
) -> tuple[float, float]:
    """Get BM25 parameters from the YAML config. Provided k_1 and b values override the config.
    Falls back to the global default if a level/stemming-specific configuration is not present.

    Args:
        level (str): The level of the index to search. Can be either "paragraph" or "paper"
        stem (bool): Whether to use stemming or not
        k_1 (float, optional): The k_1 parameter for BM25. If None, the best parameter for the given level and stemming option will be used. Defaults to None.
        b (float, optional): The b parameter for BM25. If None, the best parameter for the given level and stemming option will be used. Defaults to None.

    Returns:
        tuple[float, float]: A tuple containing the k_1 and b parameters for BM25.
    """

    level_config = CONFIG.get(level, {})
    stem_key = "stemmed" if stem else "unstemmed"

    params = level_config.get(
        stem_key,
        CONFIG.get("default", {}),
    )
    if k_1 is None:
        k_1 = params.get(
            "k_1",
            CONFIG.get("default", {}).get("k_1", 1.2),
        )
    if b is None:
        b = params.get(
            "b",
            CONFIG.get("default", {}).get("b", 0.75),
        )
    return float(k_1), float(b)


def retrieve(query: str, level: str, stem: bool, k: int, k_1: float = None, b: float = None) -> dict:
    """Retrieves documents that match the given query
    Returns a dictionary with the top-k retrieval results, sorted descending by score

    Args:
        query (str): The query to search for
        level (str): The level of the index to search. Can be either "paragraph" or "paper"
        stem (bool): Whether to use stemming or not
        k (int): The number of results to return
        k_1 (float, optional): The k_1 parameter for BM25. If None, the best parameter for the given level and stemming option will be used. Defaults to None.
        b (float, optional): The b parameter for BM25. If None, the best parameter for the given level and stemming option will be used. Defaults to None.

    Returns:
        dict: A dictionary containing document IDs as keys with their scores as values
    """

    k_1, b = get_bm25_params(
        level=level,
        stem=stem,
        k_1=k_1,
        b=b,
    )

    index = load_index(level, stem)
    return retrieve_from_index(query, stem, index, k, k_1, b)


def retrieve_from_index(query: str, stem: bool, index: dict, k: int, k_1: float, b: float) -> dict:
    """
    Retrieves documents that match the given query from the given index

    Args:
        query (str): The query to search for
        stem (bool): Whether to use stemming or not
        index (dict): The inverted index to search
        k (int): The number of results to return
        k_1 (float): The k_1 parameter for BM25
        b (float): The b parameter for BM25
    
    Returns:
        dict: A dictionary containing document IDs as keys with their scores as values
    """

    docs = {}
    average_doc_length = indexer.get_average_document_length(index)
    terms = preprocess(query, stem)
    for term in terms:
        tf_map = indexer.get_term_frequency(index, term)
        docs_with_term = len(tf_map)
        idf = math.log((indexer.get_num_documents(index) - docs_with_term + 0.5) / (docs_with_term + 0.5) + 1)
        for doc_id, count in tf_map.items():
            term_score = idf * (count * (k_1 + 1)) / (count + k_1 * (1 - b + b * indexer.get_document_length(index, doc_id) / average_doc_length))
            if doc_id not in docs:
                docs[doc_id] = 0
            docs[doc_id] += term_score

    return dict(sorted(docs.items(), key=lambda item: item[1], reverse=True)[:k])


def prepare_query(
    query: str,
    stem: bool,
    index: dict,
) -> dict:
    """
    Precompute everything from a query that does not depend on k1/b

    Args:
        query: The query to prepare
        stem: Whether to use stemming or not
        index: The inverted index to use for retrieval

    Returns:
        dict: A dictionary containing the average document length and a list of postings for each term in the query. Each posting contains the document IDs, term frequencies, document lengths, and IDF for the term.
    """

    average_doc_length = indexer.get_average_document_length(index)
    num_documents = indexer.get_num_documents(index)

    terms = preprocess(query, stem)
    postings = []

    for term in terms:
        tf_map = indexer.get_term_frequency(index, term)

        if not tf_map:
            continue

        docs_with_term = len(tf_map)

        idf = math.log(
            (num_documents - docs_with_term + 0.5)
            / (docs_with_term + 0.5)
            + 1
        )
        doc_ids = list(tf_map.keys())
        term_frequencies = np.asarray(
            [tf_map[doc_id] for doc_id in doc_ids],
            dtype=np.float64,
        )
        document_lengths = np.asarray(
            [
                indexer.get_document_length(index, doc_id)
                for doc_id in doc_ids
            ],
            dtype=np.float64,
        )
        postings.append(
            {
                "doc_ids": doc_ids,
                "tf": term_frequencies,
                "doc_lengths": document_lengths,
                "idf": idf,
            }
        )
    return {
        "average_doc_length": average_doc_length,
        "postings": postings,
    }


def retrieve_prepared(
    prepared_query: dict,
    k: int,
    k_1: float,
    b: float,
) -> dict:
    """
    Retrieve using a query prepared by prepare_query()

    Args:
        prepared_query: The query prepared by prepare_query()
        k: The number of results to return
        k_1: The k1 parameter for BM25
        b: The b parameter for BM25
    
    Returns:
        dict: A dictionary containing document IDs as keys with their scores as values
    """

    postings = prepared_query["postings"]
    average_doc_length = prepared_query["average_doc_length"]

    if not postings or average_doc_length <= 0:
        return {}

    scores = {}

    for posting in postings:
        doc_ids = posting["doc_ids"]
        tf = posting["tf"]
        doc_lengths = posting["doc_lengths"]
        idf = posting["idf"]

        denominator = (
            tf
            + k_1
            * (
                1.0
                - b
                + b * doc_lengths / average_doc_length
            )
        )
        term_scores = (
            idf
            * (tf * (k_1 + 1.0))
            / denominator
        )
        for doc_id, score in zip(doc_ids, term_scores):
            scores[doc_id] = scores.get(doc_id, 0.0) + float(score)
    return dict(
        sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:k]
    )


def prepare_queries(
    queries: list[tuple[str, str]],
    level: str,
    stem: bool,
) -> list[tuple[str, dict]]:
    """
    Prepare all queries for a particular index once.

    Args:
        queries: List of (query_id, query_text) tuples.
        level: The level of the index to search. Can be either "paragraph" or "paper"
        stem: Whether to use stemming or not

    Returns:
        List of (query_id, prepared_query) tuples.
    """

    index = load_index(level, stem)
    return [
        (
            qid,
            prepare_query(
                query,
                stem,
                index,
            ),
        )
        for qid, query in queries
    ]

def main():
    """REPL for using the retrieval system"""
    while True:
        query = input("Enter a query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        while True:
            level = input("Enter the level of the index to search (paragraph/paper): ").strip().lower()
            if level in ("paragraph", "paper"):
                break
        while True:
            stem = input("Use stemming? (y/n): ").strip().lower()
            if stem == "y":
                stem = True
                break
            elif stem == "n":
                stem = False
                break
        while True:
            k = input("Enter how many results to return: ").strip()
            try:
                k = int(k)
            except:
                continue
            if k > 0:
                break
        results = retrieve(query, level, stem, k)
        print(f"Top {k} results for query '{query}':")
        for doc_id, score in results.items():
            print(f"{doc_id}: {score:.4f}")
        print("-" * 50)


if __name__ == "__main__":
    main()
