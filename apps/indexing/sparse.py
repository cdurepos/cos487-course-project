"""Index all arXiv paragraphs and papers with the fine-tuned SPLADE model."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from sentence_transformers import SparseEncoder
from tqdm.auto import tqdm

from apps.utils.encoding import encode

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "JSON Files"
OUT = ROOT / "data" / "indexes" / "splade"
TOKENS, GROUP = 256, 16

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--limit", type=int, default=0, help="Index only N papers for testing")
args = parser.parse_args()

model = SparseEncoder(str(ROOT / "models" / "splade"))
model.max_seq_length = TOKENS
model.eval()
tok, device = model.tokenizer, model.device
D = model.get_embedding_dimension()
indexes = {name: {"ids": [], "data": [], "cols": [], "ptr": [0]}
           for name in ("paragraphs", "papers")}


def add(level, docid, vector):
    index = indexes[level]
    active = np.flatnonzero(vector)
    index["ids"].append(docid)
    index["data"].append(vector[active])
    index["cols"].append(active.astype(np.int32))
    index["ptr"].append(index["ptr"][-1] + len(active))


files = sorted(DATA.glob("*.json"))
if args.limit:
    files = files[:args.limit]
if not files:
    raise FileNotFoundError(f"No paper JSON files found in {DATA}")

with torch.inference_mode():
    for path in tqdm(files, desc="Indexing SPLADE"):
        article = json.loads(path.read_text())
        pid = article["paper_id"]
        paragraphs = article["paragraphs"]
        ids = [None, *paragraphs.keys()]
        texts = [f'{article.get("title", "")}. {article.get("abstract", "")}',
                 *paragraphs.values()]
        paper_vector = np.zeros(D, dtype=np.float32)

        for i in range(0, len(texts), GROUP):
            vectors = encode(
                model, texts[i:i + GROUP],
                pooling="max",
                batch_size=args.batch_size,
                max_tokens=TOKENS
            ).cpu().numpy()
            np.maximum(paper_vector, vectors.max(axis=0), out=paper_vector)
            for docid, vector in zip(ids[i:i + GROUP], vectors):
                if docid is not None:
                    add("paragraphs", docid, vector)
        add("papers", pid, paper_vector)

OUT.mkdir(parents=True, exist_ok=True)
for level, index in indexes.items():
    data = np.concatenate(index["data"]) if index["data"] else np.array([], dtype=np.float32)
    cols = np.concatenate(index["cols"]) if index["cols"] else np.array([], dtype=np.int32)
    matrix = sparse.csr_matrix((data, cols, index["ptr"]),
                               shape=(len(index["ids"]), D))
    sparse.save_npz(OUT / f"{level}.npz", matrix)
    (OUT / f"{level}_ids.json").write_text(json.dumps(index["ids"]))
    print(f"{level}: {matrix.shape[0]:,} documents, {matrix.nnz:,} nonzero weights")
