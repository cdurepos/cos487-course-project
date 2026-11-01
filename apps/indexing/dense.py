"""
Encodes paragraphs using the Qwen-8B model and saves the embeddings to a compressed .npz file.

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
MODEL = "Qwen/Qwen3-Embedding-8B"
EMBEDDINGS_PATH = ROOT / "data" / "indexes" / "qwen_pretrained.npz"


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=MODEL)
parser.add_argument("--embeddings-path", type=str, default=EMBEDDINGS_PATH)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--force", action="store_true", help="Force re-creation of embeddings")
args = parser.parse_args()

embedder = SentenceTransformer(args.model)


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
    for i in tqdm(range(0, len(para_texts), batch_size)):
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


if __name__ == '__main__':
    get_embeddings(args.embeddings_path, force=args.force)
