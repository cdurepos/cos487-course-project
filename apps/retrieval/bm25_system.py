"""
Contains the BM25 retrieval system
Caches indexes, so that retrieval can be done from multiple indexes without having to reload them every time.

Includes the retrieve(query, level, stem, k, k_1, b) function to retrieve documents for a given query using the BM25 algorithm.
To create a REPL command-line interface, run this file directly from the root with the command:
    python -m apps.retrieval.bm25_system
"""

import math

from apps.processing.preprocess import preprocess
import apps.retrieval.index as indexer


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


def retrieve(query: str, level: str, stem: bool, k: int, k_1: float = 1.2, b: float = 0.75) -> dict:
    """Retrieves documents that match the given query
    Returns a dictionary with the top-k retrieval results, sorted descending by score

    Args:
        query (str): The query to search for
        level (str): The level of the index to search. Can be either "paragraph" or "paper"
        stem (bool): Whether to use stemming or not
        k (int): The number of results to return
        k_1 (float): The k1 parameter for the BM25 formula
        b (float): The b parameter for the BM25 formula

    Returns:
        dict: A dictionary containing document IDs as keys with their scores as values
    """

    index = load_index(level, stem)
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
    