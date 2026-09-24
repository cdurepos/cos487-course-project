"""
Creates inverted indexes for the corpus at both the paragraph and paper levels, with and without stemming.
Writes inverted indexes to data/indexes/

Should be run as a module from directory root (i.e., cos487-course-project/) with the command:
    python -m apps.retrieval.index

Also contains functions to load the inverted index and retrieve information from it, such as term frequency and document length.
"""

import os
import json
import time
from typing import Literal

from apps.processing.preprocess import load_corpus

INDEX_DIR = os.path.join("data", "indexes")


def build_index(level: Literal["paragraph", "paper"], stem: bool):
    """Builds an inverted index for the given level

    Args:
        level (str): The level of the index to build. Can be either "paragraph" or "paper"
        stem (bool): Whether to use stemming or not
    """

    start = time.time()
    document_lengths = {}
    terms = {}
    index = {
        "document_lengths": document_lengths,
        "terms": terms,
    }
    for docid, tokens in load_corpus(level, stem):
        document_lengths[docid] = len(tokens)
        for token in tokens:
            if token not in terms:
                terms[token] = {}
            if docid not in terms[token]:
                terms[token][docid] = 0
            terms[token][docid] += 1

    index_path = f"{INDEX_DIR}/{level}_index_{'un' if not stem else ''}stemmed.json"
    with open(index_path, "w", encoding="utf-8") as file:
        json.dump(index, file, indent=4)
    print(f"Built {level} index with{'out' if not stem else ''} stemming, saved to {index_path}")
    print(f"Time taken: {(time.time() - start) / 60:.2f} minutes\n")



def load_index(level: Literal["paragraph", "paper"], stem: bool) -> dict:
    """Loads an inverted index for the given level

    Args:
        level (str): The level of the index to load. Can be either "paragraph" or "paper"
        stem (bool): Whether to use stemming or not

    Returns:
        dict: The inverted index
    """

    index_path = f"{INDEX_DIR}/{level}_index_{'un' if not stem else ''}stemmed.json"
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file {index_path} does not exist. Please build the index first.")
    
    with open(index_path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_term_frequency(index: dict, term: str) -> dict:
    """Gets the frequencies for a given term from the inverted index

    Args:
        index (dict): The inverted index
        term (str): The term to get frequencies for
    
    Returns:
        dict: A dictionary containing docID -> term count for the given term.
    """

    if term not in index["terms"]:
        return {}
    return index["terms"][term]


def get_document_length(index: dict, docid: str) -> int | None:
    """Gets the number of tokens in a given document from the inverted index

    Args:
        index (dict): The inverted index
        docid (str): The document ID to get the length for

    Returns:
        int | None: The length of the document, or None if the document is not found.
    """
    
    return index["document_lengths"].get(docid, None)


def get_num_documents(index: dict) -> int:
    """Gets the number of documents in the inverted index

    Args:
        index (dict): The inverted index

    Returns:
        int: The number of documents in the index.
    """

    return len(index["document_lengths"])


def get_average_document_length(index: dict) -> float:
    """Gets the average document length in the inverted index

    Args:
        index (dict): The inverted index

    Returns:
        float: The average document length in the index.
    """

    total_length = sum(index["document_lengths"].values())
    num_documents = len(index["document_lengths"])
    return total_length / num_documents if num_documents > 0 else 0.0


def main():
    """Main function to build the indexes for both paragraph and paper levels, with and without stemming."""
    os.makedirs(INDEX_DIR, exist_ok=True)
    for level in ("paragraph", "paper"):
        for stem in (True, False):
            build_index(level, stem)


if __name__ == "__main__":
    main()
