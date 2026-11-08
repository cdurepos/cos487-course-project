from sentence_transformers import SentenceTransformer, util
from FlagEmbedding import FlagReranker
import numpy as np
from pathlib import Path
import json
import torch
import os
from typing import Literal
import gc


ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_DIR = ROOT / "data" / "indexes"

MODEL = "Qwen/Qwen3-Embedding-0.6B"
FINETUNED_ADAPTER = ROOT / "models" / "qwen" / "best_adapter"

CROSSENCODER_MODEL = "BAAI/bge-reranker-v2-m3"
FINETUNED_CROSSENCODER = ROOT / "models" / "bge_reranker" / "best_model"
BATCH_SIZE = 32
MAX_LENGTH = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INSTRUCTION = "Given a scientific research question, retrieve paragraphs from scientific papers that are relevant to answering the question."

CORPUS = ROOT / "data" / "JSON Files"
PARAGRAPHS = {}

indexes = {
    "pretrained_paragraph": INDEX_DIR / "qwen_pretrained.npz",
    "pretrained_paper": INDEX_DIR / "qwen_pretrained_papers.npz",
    "finetuned_paragraph": INDEX_DIR / "qwen_finetuned.npz",
    "finetuned_paper": INDEX_DIR / "qwen_finetuned_papers.npz",
}

STUDY_TOPICS = ROOT / "data" / "Study.json"
TEST_TOPICS = ROOT / "data" / "Test.json"

EMBEDDINGS = {}

pretrained_model = None
finetuned_model = None
pretrained_crossencoder = None
finetuned_crossencoder = None


_prepare_query = lambda query: f"Instruct: {INSTRUCTION}\nQuery: {query}"

def search(
        query,
        training: Literal["pretrained", "finetuned"],
        level: Literal["paragraph", "paper"],
        paper_method: Literal["mean_pooling", "from_paragraphs", "crossencoder"] = None,
        paragraph_method: Literal["cosine_similarity", "crossencoder"] = None,
        crossencoder_training: Literal["pretrained", "finetuned"] = None,
        k=100
    ):
    """
    Searches for the top-k documents for a query

    Args:
        query (str): the query to search for
        training (Literal["pretrained", "finetuned"], optional): which embeddings to use
        level (Literal["paragraph", "paper"], optional): which level of embeddings to use
        paper_method (Literal["mean_pooling", "from_paragraphs", "crossencoder"], optional): which method to use for paper-level retrieval. Must be specified if level is "paper".
        paragraph_method (Literal["cosine_similarity", "crossencoder"], optional): which method to use for paragraph-level retrieval. Must be specified if level is "paragraph".
        crossencoder_training (Literal["pretrained", "finetuned"], optional): whether to use the fine-tuned or pretrained crossencoder. Must be specified if method is "crossencoder".
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
    if level == "paragraph" and paragraph_method not in ["cosine_similarity", "crossencoder"]:
        raise ValueError("paragraph_method must be either 'cosine_similarity' or 'crossencoder'")
    if level == "paper" and paper_method not in ["mean_pooling", "from_paragraphs", "crossencoder"]:
        raise ValueError("paper_method must be 'mean_pooling', 'from_paragraphs', or 'crossencoder'")

    if level == "paper":
        method = paper_method
    else:
        method = paragraph_method
    if method == "crossencoder" and crossencoder_training not in ["pretrained", "finetuned"]:
        raise ValueError("crossencoder_training must be either 'pretrained' or 'finetuned' if method is 'crossencoder'")

    global pretrained_model, finetuned_model, pretrained_crossencoder, finetuned_crossencoder
    model = None
    crossencoder = None
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
    if method == "crossencoder":
        _load_paragraphs()
        if crossencoder_training == "pretrained":
            if pretrained_crossencoder is None:
                pretrained_crossencoder = FlagReranker(
                    CROSSENCODER_MODEL,
                    use_fp16=True,
                    devices=[DEVICE]
                )
            crossencoder = pretrained_crossencoder
        else:
            if finetuned_crossencoder is None:
                finetuned_crossencoder = FlagReranker(
                    FINETUNED_CROSSENCODER,
                    use_fp16=True,
                    devices=[DEVICE]
                )
            crossencoder = finetuned_crossencoder

    embeddings = _get_embeddings(training=training, level="paper" if (level=="paper" and method=="mean_pooling") else "paragraph")
    doc_ids = embeddings["ids"]
    doc_embeddings = embeddings["embeddings"]

    qwen_query = _prepare_query(query)
    query_embedding = model.encode(qwen_query, convert_to_tensor=True)
    query_embedding = query_embedding.float()

    top_k = k * 4 if (level=="paper" and method!="mean_pooling") else k
    while True:
        doc_results = {}
        hits = util.semantic_search(query_embedding, doc_embeddings, top_k=top_k)
        hits = hits[0]
        for hit in hits:
            index = hit['corpus_id']
            doc_id = doc_ids[index]
            score = hit['score']
            doc_results[doc_id] = score

        if (level=="paragraph" and method=="cosine_similarity") or (level=="paper" and method=="mean_pooling"):
            return list(sorted(doc_results.items(), key=lambda item: item[1], reverse=True))
        if level == "paper":
            if len(set(key.split("_")[0] for key in doc_results.keys())) < k:
                top_k *= 2
                continue
            results = {}
            paper_ids = set()
            for para_id, para_score in sorted(doc_results.items(), key=lambda item: item[1], reverse=True):
                paper_id = para_id.split("_")[0]
                if paper_id not in paper_ids:
                    paper_ids.add(paper_id)
                    results[para_id] = para_score
                if len(paper_ids) >= k:
                    break
            if method == "from_paragraphs":
                return list(sorted({para_id.split("_")[0]: score for para_id, score in results.items()}.items(), key=lambda item: item[1], reverse=True))
            doc_results = results

        scores = crossencoder.compute_score(list((query, PARAGRAPHS[para_id]) for para_id in doc_results.keys()), batch_size=BATCH_SIZE, max_length=MAX_LENGTH)
        results = {para_id: float(score) for para_id, score in zip(list(doc_results.keys()), scores)}
        if level == "paragraph":
            return list(sorted(results.items(), key=lambda item: item[1], reverse=True))
        return list(sorted({para_id.split("_")[0]: score for para_id, score in results.items()}.items(), key=lambda item: item[1], reverse=True))


def delete_models_and_indexes():
    global pretrained_model, finetuned_model, pretrained_crossencoder, finetuned_crossencoder
    pretrained_model = None
    finetuned_model = None
    pretrained_crossencoder = None
    finetuned_crossencoder = None

    EMBEDDINGS.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_paragraphs():
    """Loads the paragraph corpus into memory"""
    global PARAGRAPHS
    if PARAGRAPHS:
        return
    print("Loading paragraph corpus...")
    for file in os.listdir(CORPUS):
        if file.endswith(".json"):
            with open(CORPUS / file, "r") as file:
                data = json.load(file)
            if "paragraphs" in data:
                for para_id, text in data["paragraphs"].items():
                    PARAGRAPHS[para_id] = text


def _get_embeddings(training: Literal["pretrained", "finetuned"], level: Literal["paragraph", "paper"]):
    """Loads embeddings from embeddings file. Returns a dictionary with keys "ids" and "embeddings" """
    global EMBEDDINGS
    key = f"{training}_{level}"
    if key not in EMBEDDINGS:
        filepath = indexes[key]
        data = np.load(filepath)
        ids = [str(id) for id in list(data["ids"])]
        embeddings = data["embeddings"]
        EMBEDDINGS[key] = {
            "ids": ids,
            "embeddings": embeddings
        }
        print(f"Loaded {len(ids)} embeddings from {filepath}")
    return EMBEDDINGS[key]
