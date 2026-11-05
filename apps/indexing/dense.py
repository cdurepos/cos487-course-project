"""
Encodes paragraphs using the Qwen-8B model and saves the embeddings to a compressed .npz file.
Also encodes paper-level embeddings by mean pooling paragraph embeddings, title embedding, and abstract embedding.

Usage:
    python -m apps.indexing.dense
"""


from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
from pathlib import Path
import argparse
import json
import os
import torch


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "JSON Files"
MODEL = "Qwen/Qwen3-Embedding-0.6B"
FINETUNED_ADAPTER = ROOT / "models" / "qwen" / "best_adapter"

EMBEDDINGS_PATH = ROOT / "data" / "indexes" / "qwen_pretrained.npz"
EMBEDDINGS_PATH_PAPERS = ROOT / "data" / "indexes" / "qwen_pretrained_papers.npz"

FINETUNED_EMBEDDINGS_PATH = ROOT / "data" / "indexes" / "qwen_finetuned.npz"
FINETUNED_EMBEDDINGS_PATH_PAPERS = ROOT / "data" / "indexes" / "qwen_finetuned_papers.npz"


parser = argparse.ArgumentParser()
parser.add_argument("--pretrained", action="store_true")
parser.add_argument("--finetuned", action="store_true")
parser.add_argument("--paragraphs", action="store_true")
parser.add_argument("--papers", action="store_true")
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--force", action="store_true", help="Force re-creation of embeddings")
args = parser.parse_args()

embedder = SentenceTransformer(MODEL)
print(f"Loaded model: {MODEL} on device: {embedder.device}")


def load_paragraphs():
    """
    Load paragraphs from the JSON files in the data directory.
    Returns a dictionary mapping paragraph IDs to their text.
    """
    paragraphs = {}
    for json_file in os.listdir(DATA):
        if json_file.endswith(".json"):
            with open(DATA / json_file, "r") as f:
                paper = json.load(f)
                if "paragraphs" not in paper:
                    continue
                for id, text in paper["paragraphs"].items():
                    paragraphs[id] = text
    return paragraphs


def encode_batch(batch, batch_ids, embedder):
    """
    Encode a batch of texts using the provided embedder.
    Returns a numpy array of embeddings.
    """
    try:
        embeddings = embedder.encode(batch,
                                     convert_to_numpy=True, 
                                     normalize_embeddings=True,
                                     show_progress_bar=False)
        return embeddings, batch_ids
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return encode_separately(batch, batch_ids, embedder)


def encode_separately(batch, batch_ids, embedder):
    """
    Encode each text in the batch separately to avoid memory issues.
    Returns a numpy array of embeddings.
    """
    embeddings = []
    ids = []
    for text, id in zip(batch, batch_ids):
        try:
            embedding = embedder.encode(text, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
            embeddings.append(embedding)
            ids.append(id)
        except Exception:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            embedding = encode_paragraph_chunk(text, embedder)
            if embedding is not None:
                embeddings.append(embedding)
                ids.append(id)
    return np.array(embeddings), ids


def encode_paragraph_chunk(text, embedder, num_chunks=2, depth=1):
    """
    Encode a single paragraph by splitting it into smaller chunks until it fits in memory
    Returns the mean pooling of the chunk embeddings.
    """
    if depth > 10:
        print(f"Failed to encode paragraph after 10 attempts: {text[:100]}...")
        return None
    try:
        tokenizer = embedder.tokenizer
        token_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        chunks = np.array_split(token_ids, num_chunks)
        chunk_embeddings = []
        for chunk_ids in chunks:
            chunk_text = tokenizer.decode(chunk_ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
            embedding = embedder.encode(chunk_text, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
            chunk_embeddings.append(embedding)
        embedding = np.mean(chunk_embeddings, axis=0)
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return None
        return embedding / norm
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return encode_paragraph_chunk(text, embedder, num_chunks=num_chunks+1, depth=depth+1)


def get_embeddings(save_path, force=False):
    """
    This method creates the embeddings for the answers
    if a file does not already exist at the given filepath
    """
    if not force and Path(save_path).is_file():
        print(f"Embeddings found at {save_path}.")
        return
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    print("Reading corpus...")
    paragraphs = load_paragraphs()
    para_ids = list(paragraphs.keys())
    para_texts = list(paragraphs.values())

    batch_size = args.batch_size

    all_embeddings = []
    all_ids = []
    for i in tqdm(range(0, len(para_texts), batch_size), desc="Encoding paragraphs"):
        batch = para_texts[i:i+batch_size]
        batch_ids = para_ids[i:i+batch_size]
        embeddings, ids = encode_batch(batch, batch_ids, embedder)
        all_embeddings.append(embeddings)
        all_ids += ids
    all_embeddings = np.vstack(all_embeddings)
    np.savez_compressed(
        save_path,
        ids=np.array(all_ids),
        embeddings=all_embeddings
    )
    print(f"Wrote embeddings to {save_path}.")


def encode_text(text, embedder):
    try:
        return embedder.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return encode_paragraph_chunk(text, embedder)


def encode_paper(paper, paragraph_embeddings, embedder):
    """
    Generate a paper-level embedding by mean pooling its paragraph
    embeddings, title embedding, and abstract embedding.

    If the abstract is already present as a paragraph, its embedding
    is not added again.
    """
    embeddings = []
    paragraphs = paper.get("paragraphs", {})
    
    for paragraph_id in paragraphs:
        if paragraph_id in paragraph_embeddings:
            embeddings.append(paragraph_embeddings[paragraph_id])

    title = paper.get("title", "").strip()
    if title:
        title_embedding = embedder.encode(
            title,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        embeddings.append(title_embedding)

    abstract = paper.get("abstract", "").strip()
    if abstract:
        abstract_is_paragraph = any(
            paragraph.strip() == abstract
            for paragraph in paragraphs.values()
            if isinstance(paragraph, str)
        )
        if not abstract_is_paragraph:
            abstract_embedding = encode_text(abstract, embedder)
            embeddings.append(abstract_embedding)
    if not embeddings:
        return None

    paper_embedding = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(paper_embedding)
    if norm == 0:
        return None
    return paper_embedding / norm


def get_paper_embeddings(paragraph_embeddings_path, save_path, force=False):
    """
    Generate paper-level embeddings using previously computed
    paragraph embeddings.

    Saves a compressed .npz containing paper IDs and embeddings.
    """
    if not force and Path(save_path).is_file():
        print(f"Paper embeddings found at {save_path}.")
        return
    if not Path(paragraph_embeddings_path).is_file():
        print(f"Paragraph embeddings not found at {paragraph_embeddings_path}.")
        return
    data = np.load(paragraph_embeddings_path, allow_pickle=True)
    paragraph_embeddings = {
        id: embedding
        for id, embedding in zip(data["ids"], data["embeddings"])
    }
    paper_ids = []
    paper_embeddings = []
    print("Generating paper-level embeddings...")

    for json_file in tqdm(os.listdir(DATA)):
        if not json_file.endswith(".json"):
            continue
        with open(DATA / json_file, "r") as f:
            paper = json.load(f)
        embedding = encode_paper(
            paper,
            paragraph_embeddings,
            embedder
        )
        if embedding is not None:
            paper_id = json_file.removesuffix(".json")
            paper_ids.append(paper_id)
            paper_embeddings.append(embedding)
    if not paper_embeddings:
        print("No paper embeddings generated.")
        return
    np.savez_compressed(
        save_path,
        ids=np.array(paper_ids),
        embeddings=np.vstack(paper_embeddings)
    )
    print(f"Wrote paper embeddings to {save_path}.")


if __name__ == '__main__':
    force = args.force
    finetuned = args.finetuned
    pretrained = args.pretrained
    paragraphs = args.paragraphs
    papers = args.papers

    if not finetuned and not pretrained:
        finetuned = True
        pretrained = True
    if not paragraphs and not papers:
        paragraphs = True
        papers = True

    if pretrained:
        if paragraphs:
            get_embeddings(EMBEDDINGS_PATH, force=force)
        if papers:
            get_paper_embeddings(EMBEDDINGS_PATH, EMBEDDINGS_PATH_PAPERS, force=force)
    if finetuned:
        embedder.load_adapter(FINETUNED_ADAPTER)
        embedder.eval()
        print(f"Loaded finetuned adapter from {FINETUNED_ADAPTER}.")
        if paragraphs:
            get_embeddings(FINETUNED_EMBEDDINGS_PATH, force=force)
        if papers:
            get_paper_embeddings(FINETUNED_EMBEDDINGS_PATH, FINETUNED_EMBEDDINGS_PATH_PAPERS, force=force)
