"""
TF-IDF retrieval system (the vector space model, ranked by cosine similarity)
Caches indexes (and their document norms), so retrieval can be done from multiple indexes without reloading them every time.

Includes the retrieve(query, level, stem, k) function to retrieve documents for a given query using TF-IDF.
Uses the same inverted indexes that index.py builds, so nothing new has to be built first.
To create a REPL command-line interface, run this file directly from the root with the command:
    python -m apps.retrieval.tfidf

"""

import math
from collections import Counter

from apps.processing.preprocess import preprocess
import apps.retrieval.index as indexer


# same idea as bm25.py: keep loaded indexes around so the second query doesn't reload a giant JSON file
indexes = {}

# document norms (vector lengths) for each index, computed once and then reused.
# they only depend on the index, not on the query, so there's no point redoing them every search
doc_norms = {}

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


def tf_weight(count: int) -> float:
    """Turns a raw term count into a log-scaled tf weight

    1 time -> 1.0, 2 times -> 1.69, 10 times -> 3.30, 100 times -> 5.61
    so more mentions still help, just less and less each time (diminishing returns)

    Args:
        count (int): How many times the term shows up (always at least 1 here)

    Returns:
        float: The tf weight
    """

    return 1 + math.log(count)


def idf_weight(num_documents: int, docs_with_term: int) -> float:
    """Gets the idf (inverse document frequency) weight for a term

    rare term (in 10 of 1000 docs) -> ln(100) = 4.6, a big deal
    common term (in 900 of 1000 docs) -> ln(1.11) = 0.1, barely matters
    term in every single doc -> ln(1) = 0, doesn't help tell docs apart at all

    Args:
        num_documents (int): N, the total number of documents in the index
        docs_with_term (int): df, how many documents contain the term

    Returns:
        float: The idf weight (0 if the term isn't in any document)
    """

    if docs_with_term == 0:
        return 0.0
    return math.log(num_documents / docs_with_term)


def load_doc_norms(level: str, stem: bool) -> dict:
    """Gets the length (norm) of every document's tf-idf vector, computing it the first time

    Heads up: this is the slow part. It walks through every (term, document) pair in the index once,
    which for the paragraph index is a LOT of pairs, so the first search on an index takes a while.
    After that it's cached and every other search is quick.

    Args:
        level (str): The level of the index. Can be either "paragraph" or "paper"
        stem (bool): Whether to use stemming or not

    Returns:
        dict: A dictionary mapping document IDs to the length of their tf-idf vector
    """

    global doc_norms
    id = index_id(level, stem)
    if id in doc_norms:
        return doc_norms[id]

    print("Computing document norms (only happens once per index)...")
    index = load_index(level, stem)
    num_documents = indexer.get_num_documents(index)

    # the norm of a vector is sqrt(w1^2 + w2^2 + w3^2 + ...), one w for every word in the doc.
    # the index is organized word -> docs (not doc -> words), so we can't grab one doc's words directly.
    # instead we go word by word and add each word's squared weight onto the running total for every doc it's in.
    # by the end, each doc's total has every one of its words in it, then we just square root it
    squared_sums = {}
    for term, tf_map in index["terms"].items():
        idf = idf_weight(num_documents, len(tf_map))
        for doc_id, count in tf_map.items():
            weight = tf_weight(count) * idf
            squared_sums[doc_id] = squared_sums.get(doc_id, 0) + weight * weight

    norms = {}
    for doc_id, total in squared_sums.items():
        norms[doc_id] = math.sqrt(total)

    doc_norms[id] = norms
    return norms


def retrieve(query: str, level: str, stem: bool, k: int) -> dict:
    """Retrieves documents that match the given query
    Returns a dictionary with the top-k retrieval results, sorted descending by score

    Args:
        query (str): The query to search for
        level (str): The level of the index to search. Can be either "paragraph" or "paper"
        stem (bool): Whether to use stemming or not
        k (int): The number of results to return

    Returns:
        dict: A dictionary containing document IDs as keys with their scores (between 0 and 1) as values
    """

    index = load_index(level, stem)
    norms = load_doc_norms(level, stem)
    num_documents = indexer.get_num_documents(index)

    # clean the query the exact same way the documents were cleaned (same stem flag too),
    # otherwise query words and index words won't line up
    terms = preprocess(query, stem)

    # count each query word, so "neural neural networks" -> {"neural": 2, "networks": 1}
    query_counts = Counter(terms)

    # build the query's tf-idf vector, keeping only words that are actually in the index
    # (a word that's in no document can't match anything, so it's useless here)
    query_weights = {}
    for term, count in query_counts.items():
        tf_map = indexer.get_term_frequency(index, term)
        idf = idf_weight(num_documents, len(tf_map))
        if idf > 0:
            query_weights[term] = tf_weight(count) * idf

    # nothing usable in the query (all stopwords, typos, or words in every doc) -> no results
    if not query_weights:
        return {}

    query_norm = math.sqrt(sum(weight * weight for weight in query_weights.values()))

    # dot product, one word at a time: for every doc that has the word,
    # add (query weight for the word) * (doc weight for the word) onto that doc's running total.
    # docs that share no words with the query never show up here, which is why this is fast
    docs = {}
    for term, query_weight in query_weights.items():
        tf_map = indexer.get_term_frequency(index, term)
        idf = idf_weight(num_documents, len(tf_map))
        for doc_id, count in tf_map.items():
            doc_weight = tf_weight(count) * idf
            if doc_id not in docs:
                docs[doc_id] = 0
            docs[doc_id] += query_weight * doc_weight

    # turn each dot product into a cosine similarity by dividing by both vector lengths.
    # this is what stops long docs from winning just because they're long
    for doc_id in docs:
        docs[doc_id] = docs[doc_id] / (norms[doc_id] * query_norm)

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
            except ValueError:
                continue
            if k > 0:
                break
        results = retrieve(query, level, stem, k)
        if not results:
            print("No documents matched that query.")
        else:
            print(f"Top {k} results for query '{query}':")
            for doc_id, score in results.items():
                print(f"{doc_id}: {score:.4f}")
        print("-" * 50)


if __name__ == "__main__":
    main()
