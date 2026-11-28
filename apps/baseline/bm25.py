"""BM25 scoring for one query against an index that the caller supplies."""

import math

from apps.processing.preprocess import preprocess
import apps.baseline.index as indexer


def retrieve(query: str, index: dict, k: int, k_1: float, b: float, stem: bool = True) -> dict:
    """Score one query against an inverted index.

    Args:
        query (str): The query to search for
        index (dict): The inverted index to search
        k (int): The number of results to return
        k_1 (float): The k_1 parameter for BM25
        b (float): The b parameter for BM25
        stem (bool, optional): Stem query tokens before lookup. Defaults to True.

    Returns:
        dict: Document IDs mapped to BM25 scores, highest score first.
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

