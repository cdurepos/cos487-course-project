"""
Retrieves paragraphs and papers using the Qwen-8B model
Running the file directly will run the retrieval for both pretrained and finetuned embeddings, 
for both paragraph-level and paper-level retrieval.

Usage:
    python -m apps.retrieval.dense

Also contains the search function that can be imported and used in other scripts.
"""


from sentence_transformers import SentenceTransformer, util
import numpy as np
from pathlib import Path
import torch
from typing import Literal
import argparse
import json
import time
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent.parent

MODEL = "Qwen/Qwen3-Embedding-0.6B"
FINETUNED_ADAPTER = ROOT / "models" / "qwen" / "best_adapter"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INSTRUCTION = (
    "Given a scientific research question, retrieve scientific paper "
    "paragraphs that are relevant to answering the question."
)

embeddings_and_outputs = {
    "pretrained_paragraph": (
        ROOT / "data" / "indexes" / "qwen_pretrained.npz",
        ROOT / "data" / "runs" / "qwen_pretrained_paragraph_results_study.tsv",
        ROOT / "data" / "runs" / "qwen_pretrained_paragraph_results_test.tsv"
    ),
    "pretrained_paper": (
        ROOT / "data" / "indexes" / "qwen_pretrained_papers.npz",
        ROOT / "data" / "runs" / "qwen_pretrained_paper_results_study.tsv",
        ROOT / "data" / "runs" / "qwen_pretrained_paper_results_test.tsv"
    ),
    "finetuned_paragraph": (
        ROOT / "data" / "indexes" / "qwen_finetuned.npz",
        ROOT / "data" / "runs" / "qwen_finetuned_paragraph_results_study.tsv",
        ROOT / "data" / "runs" / "qwen_finetuned_paragraph_results_test.tsv"
    ),
    "finetuned_paper": (
        ROOT / "data" / "indexes" / "qwen_finetuned_papers.npz",
        ROOT / "data" / "runs" / "qwen_finetuned_paper_results_study.tsv",
        ROOT / "data" / "runs" / "qwen_finetuned_paper_results_test.tsv"
    ),
}

STUDY_TOPICS = ROOT / "data" / "Study.json"
TEST_TOPICS = ROOT / "data" / "Test.json"

EMBEDDINGS = {}

pretrained_model = None
finetuned_model = None


def get_embeddings(training: Literal["pretrained", "finetuned"], level: Literal["paragraph", "paper"], filepath):
    """Loads embeddings from embeddings file

    Args:
        training (Literal["pretrained", "finetuned"]): which embeddings to load
        level (Literal["paragraph", "paper"]): which level of embeddings to load
        filepath (.npz): dictionary with id: embedding

    Returns:
        dict: dictionary with ids: list, embeddings: numpy ndarray
    """

    if training not in ["pretrained", "finetuned"]:
        raise ValueError("training must be either 'pretrained' or 'finetuned'")
    if level not in ["paragraph", "paper"]:
        raise ValueError("level must be either 'paragraph' or 'paper'")

    global EMBEDDINGS
    data = np.load(filepath)
    para_ids = [str(id) for id in list(data["ids"])]
    para_embeddings = data["embeddings"]
    print(f"Loaded {len(para_ids)} {level} embeddings from {filepath}")

    EMBEDDINGS[filepath] = {
        "ids": para_ids,
        "embeddings": para_embeddings
    }


def prepare_query(query):
    return f"Instruct: {INSTRUCTION}\nQuery: {query}"


def search(query, training: Literal["pretrained", "finetuned"], level: Literal["paragraph", "paper"], k=10):
    """Searches for the top-k answers for a query based on cosine similarity

    Args:
        query (str): the query to search for
        training (Literal["pretrained", "finetuned"], optional): which embeddings to use
        level (Literal["paragraph", "paper"], optional): which level of embeddings to use
        k (int, optional): number of results to return. Defaults to 10.

    Returns:
        list[Tuple[str, float]]: the ids of the top-k documents and their relevance scores, sorted by relevance descending
    """

    if k < 1:
        raise ValueError("k must be greater than 0")
    if training not in ["pretrained", "finetuned"]:
        raise ValueError("training must be either 'pretrained' or 'finetuned'")
    if level not in ["paragraph", "paper"]:
        raise ValueError("level must be either 'paragraph' or 'paper'")
    
    if embeddings_and_outputs[f"{training}_paragraph"][0] not in EMBEDDINGS:
        get_embeddings(training=training, level="paragraph", filepath=embeddings_and_outputs[f"{training}_paragraph"][0])
    if level == "paper":
        if embeddings_and_outputs[f"{training}_paper"][0] not in EMBEDDINGS:
            get_embeddings(training=training, level="paper", filepath=embeddings_and_outputs[f"{training}_paper"][0])

    embeddings = EMBEDDINGS[embeddings_and_outputs[f"{training}_paragraph"][0]]
    para_ids = embeddings["ids"]
    para_embeddings = embeddings["embeddings"]

    global pretrained_model, finetuned_model
    model = None
    if training == "pretrained":
        if pretrained_model is None:
            pretrained_model = SentenceTransformer(MODEL, device=DEVICE)
            pretrained_model.eval()
        model = pretrained_model
    elif training == "finetuned":
        if finetuned_model is None:
            finetuned_model = SentenceTransformer(MODEL, device=DEVICE)
            finetuned_model.load_adapter(FINETUNED_ADAPTER)
            finetuned_model.eval()
        model = finetuned_model

    query = prepare_query(query)
    query_embedding = model.encode(query, convert_to_tensor=True)
    query_embedding = query_embedding.float()

    para_results = {}
    hits = util.semantic_search(query_embedding, para_embeddings, top_k=k)
    hits = hits[0]
    for hit in hits:
        index = hit['corpus_id']
        para_id = para_ids[index]
        score = hit['score']
        para_results[para_id] = score
    para_results = list(sorted(para_results.items(), key=lambda item: item[1], reverse=True))[:k]

    if level == "paragraph":
        return para_results

    paper_results = {}
    embeddings = EMBEDDINGS[embeddings_and_outputs[f"{training}_paper"][0]]
    paper_ids = embeddings["ids"]
    paper_embeddings = embeddings["embeddings"]
    hits = util.semantic_search(query_embedding, paper_embeddings, top_k=k)
    hits = hits[0]
    for hit in hits:
        index = hit['corpus_id']
        paper_id = paper_ids[index]
        score = hit['score']
        paper_results[paper_id] = score
    paper_results = list(sorted(paper_results.items(), key=lambda item: item[1], reverse=True))[:k]

    merged = {}
    for para_id, para_score in para_results:
        paper_id = para_id.split("_")[0]
        if paper_id not in merged:
            merged[paper_id] = para_score
        else:
            merged[paper_id] = max(merged[paper_id], para_score)
    for paper_id, paper_score in paper_results:
        if paper_id not in merged:
            merged[paper_id] = paper_score
        else:
            merged[paper_id] += paper_score
    return list(sorted(merged.items(), key=lambda item: item[1], reverse=True))[:k]


def load_topics(file_path):
    """
    Loads topics from a JSON file
    Returns a dictionary mapping topic IDs to their text
    """
    with open(file_path, "r") as f:
        topics = json.load(f)
    return {str(topic["query_id"]): topic["query"] for topic in topics}


def write_results(results, output_file, run_name):
    """
    Writes the search results to a TSV file in TREC format
    Results is a dictionary mapping topic IDs to a list of (doc_id, score) tuples
    """
    with open(output_file, "w") as f:
        for topic_id, query_results in results.items():
            for rank, (doc_id, score) in enumerate(query_results, 1):
                f.write(f"{topic_id}\tQ0\t{doc_id}\t{rank}\t{score}\t{run_name}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", action="store_true", help="Run search with pretrained embeddings")
    parser.add_argument("--finetuned", action="store_true", help="Run search with finetuned embeddings")
    parser.add_argument("--paragraphs", action="store_true", help="Retrieve paragraph-level results instead of paper-level results")
    parser.add_argument("--papers", action="store_true", help="Retrieve paper-level results instead of paragraph-level results")
    args = parser.parse_args()

    pretrained = args.pretrained
    finetuned = args.finetuned
    if not args.pretrained and not args.finetuned:
        pretrained = True
        finetuned = True

    paragraphs = args.paragraphs
    papers = args.papers
    if not args.paragraphs and not args.papers:
        paragraphs = True
        papers = True

    study_topics = load_topics(STUDY_TOPICS)
    test_topics = load_topics(TEST_TOPICS)
    if pretrained:
        if paragraphs:
            results = {}
            for i, topics in enumerate([study_topics, test_topics], 1):
                output_file = embeddings_and_outputs["pretrained_paragraph"][i]
                print(f"Searching with pretrained paragraph embeddings, saving to {output_file}...")
                for j, (topic_id, query) in tqdm(enumerate(topics.items()), total=len(topics), desc="Searching"):
                    if j == 0:
                        results[topic_id] = search(query, training="pretrained", level="paragraph", k=100)
                        start_time = time.perf_counter()
                        continue
                    results[topic_id] = search(query, training="pretrained", level="paragraph", k=100)
                total_time = time.perf_counter() - start_time
                print(f"Total search time (excluding first query): {total_time:.2f} seconds")
                average_time = total_time / (len(topics) - 1)
                print(f"Average search time: {average_time:.4f} seconds/query")
                write_results(results, output_file, "pretrained_paragraph")
        if papers:
            results = {}
            for i, topics in enumerate([study_topics, test_topics], 1):
                output_file = embeddings_and_outputs["pretrained_paper"][i]
                print(f"Searching with pretrained paper embeddings, saving to {output_file}...")
                for j, (topic_id, query) in tqdm(enumerate(topics.items()), total=len(topics), desc="Searching"):
                    if j == 0:
                        results[topic_id] = search(query, training="pretrained", level="paper", k=100)
                        start_time = time.perf_counter()
                        continue
                    results[topic_id] = search(query, training="pretrained", level="paper", k=100)
                total_time = time.perf_counter() - start_time
                print(f"Total search time (excluding first query): {total_time:.2f} seconds")
                average_time = total_time / (len(topics) - 1)
                print(f"Average search time: {average_time:.4f} seconds/query")
                write_results(results, output_file, "pretrained_paper")
    if finetuned:
        if paragraphs:
            results = {}
            for i, topics in enumerate([study_topics, test_topics], 1):
                output_file = embeddings_and_outputs["finetuned_paragraph"][i]
                print(f"Searching with finetuned paragraph embeddings, saving to {output_file}...")
                for j, (topic_id, query) in tqdm(enumerate(topics.items()), total=len(topics), desc="Searching"):
                    if j == 0:
                        results[topic_id] = search(query, training="finetuned", level="paragraph", k=100)
                        start_time = time.perf_counter()
                        continue
                    results[topic_id] = search(query, training="finetuned", level="paragraph", k=100)
                total_time = time.perf_counter() - start_time
                print(f"Total search time (excluding first query): {total_time:.2f} seconds")
                average_time = total_time / (len(topics) - 1)
                print(f"Average search time: {average_time:.4f} seconds/query")
                write_results(results, output_file, "finetuned_paragraph")
        if papers:
            results = {}
            for i, topics in enumerate([study_topics, test_topics], 1):
                output_file = embeddings_and_outputs["finetuned_paper"][i]
                print(f"Searching with finetuned paper embeddings, saving to {output_file}...")
                for j, (topic_id, query) in tqdm(enumerate(topics.items()), total=len(topics), desc="Searching"):
                    if j == 0:
                        results[topic_id] = search(query, training="finetuned", level="paper", k=100)
                        start_time = time.perf_counter()
                        continue
                    results[topic_id] = search(query, training="finetuned", level="paper", k=100)
                total_time = time.perf_counter() - start_time
                average_time = total_time / (len(topics) - 1)
                print(f"Average search time: {average_time:.4f} seconds/query")
                print(f"Total search time (excluding first query): {total_time:.2f} seconds")
                write_results(results, output_file, "finetuned_paper")


if __name__ == "__main__":
    main()
