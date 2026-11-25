"""Search paragraph and paper indexes built from Qwen3-Embedding-0.6B."""

from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
from typing import Literal
import gc

from apps.utils import MODEL_DIR, INDEX_MAP


MODEL = "Qwen/Qwen3-Embedding-0.6B"
FINETUNED_ADAPTER = MODEL_DIR / "finetuned_qwen"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INSTRUCTION = "Given a scientific research question, retrieve paragraphs from scientific papers that are relevant to answering the question."

EMBEDDINGS = {}

pretrained_model = None
finetuned_model = None


_prepare_query = lambda query: f"Instruct: {INSTRUCTION}\nQuery: {query}"


def search(
        query,
        training: Literal["pretrained", "finetuned"],
        level: Literal["paragraph", "paper"],
        paper_method: Literal["mean_pooling", "from_paragraphs"] = None,
        k=100
    ):
    """
    Searches for the top-k documents for a query

    Args:
        query (str): the query to search for
        training (Literal["pretrained", "finetuned"]): which embeddings to use
        level (Literal["paragraph", "paper"]): which level of embeddings to use
        paper_method (Literal["mean_pooling", "from_paragraphs"], optional): which method to use for paper-level retrieval. Must be specified if level is "paper".
        k (int, optional): number of results to return. Defaults to 100.

    Returns:
        list[Tuple[str, float]]: the ids of the top-k documents and their relevance scores, sorted by relevance descending
    """

    if k < 1:
        raise ValueError("k must be greater than 0")
    if training not in ["pretrained", "finetuned"]:
        raise ValueError("training must be either 'pretrained' or 'finetuned'")
    if level not in ["paragraph", "paper"]:
        raise ValueError("level must be either 'paragraph' or 'paper'")
    if level == "paper" and paper_method not in ["mean_pooling", "from_paragraphs"]:
        raise ValueError("paper_method must be 'mean_pooling' or 'from_paragraphs'")

    global pretrained_model, finetuned_model
    if training == "pretrained":
        if pretrained_model is None:
            pretrained_model = SentenceTransformer(MODEL, device=DEVICE)
            pretrained_model.eval()
        model = pretrained_model
    else:
        if finetuned_model is None:
            finetuned_model = SentenceTransformer(MODEL, device=DEVICE)
            finetuned_model.load_adapter(FINETUNED_ADAPTER)
            finetuned_model.eval()
        model = finetuned_model

    embeddings = _get_embeddings(
        training=training,
        level="paper" if (level == "paper" and paper_method == "mean_pooling") else "paragraph",
    )
    doc_ids = embeddings["ids"]
    doc_embeddings = embeddings["embeddings"]

    qwen_query = _prepare_query(query)
    query_embedding = model.encode(qwen_query, convert_to_tensor=True)
    query_embedding = query_embedding.float()

    top_k = k * 4 if (level == "paper" and paper_method == "from_paragraphs") else k
    while True:
        doc_results = {}
        hits = util.semantic_search(query_embedding, doc_embeddings, top_k=top_k)
        hits = hits[0]
        for hit in hits:
            index = hit["corpus_id"]
            doc_id = doc_ids[index]
            score = hit["score"]
            doc_results[doc_id] = score

        if level == "paragraph" or paper_method == "mean_pooling":
            return list(sorted(doc_results.items(), key=lambda item: item[1], reverse=True))

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
        return list(sorted(
            {para_id.split("_")[0]: score for para_id, score in results.items()}.items(),
            key=lambda item: item[1], reverse=True,
        ))


def delete_models_and_indexes():
    global pretrained_model, finetuned_model
    pretrained_model = None
    finetuned_model = None

    EMBEDDINGS.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_embeddings(training: Literal["pretrained", "finetuned"], level: Literal["paragraph", "paper"]):
    """Loads embeddings from embeddings file. Returns a dictionary with keys "ids" and "embeddings" """
    global EMBEDDINGS
    key = f"{training}_{level}"
    if key not in EMBEDDINGS:
        filepath = INDEX_MAP[key]
        data = np.load(filepath)
        ids = [str(id) for id in list(data["ids"])]
        embeddings = data["embeddings"]
        EMBEDDINGS[key] = {
            "ids": ids,
            "embeddings": embeddings
        }
        print(f"Loaded {len(ids)} embeddings from {filepath}")
    return EMBEDDINGS[key]
