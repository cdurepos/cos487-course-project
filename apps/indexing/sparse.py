
"""Index arXiv papers and paragraphs with resumable SPLADE CSR shards."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SparseEncoder
from tqdm.auto import tqdm

from apps.utils.encoding import encode
from apps.utils import sharding

ROOT = Path(__file__).resolve().parents[2]
DATA, MODEL = ROOT / "data" / "JSON Files", ROOT / "models" / "splade"
TOKENS, GROUP = 256, 16

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--interval", type=int, default=100)
parser.add_argument("--limit", type=int, default=0)
args = parser.parse_args()
if args.batch_size < 1 or args.interval < 1 or args.limit < 0:
    parser.error("Invalid batch size, interval or limit")

OUT = ROOT / "data" / "indexes" / (
    f"splade_test_{args.limit}" if args.limit else "splade"
)
files = sorted(DATA.glob("*.json"))
if args.limit:
    files = files[:args.limit]
if not files:
    raise FileNotFoundError(f"No paper JSON files found in {DATA}")

spec = {
    "corpus": [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in files],
    "model": [
        (str(p.relative_to(MODEL)), p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(MODEL.rglob("*")) if p.is_file()
        and p.suffix in (".bin", ".safetensors", ".json", ".txt", ".model")
        and p.name != "loss_history.json"
    ],
    "tokens": TOKENS, "interval": args.interval,
}
signature = hashlib.sha256(json.dumps(spec).encode()).hexdigest()
state = sharding.resume(OUT, files, signature, args.interval)

model = SparseEncoder(str(MODEL))
model.max_seq_length = TOKENS
model.eval()
dimension = model.get_embedding_dimension()
buffers = sharding.init()

with torch.inference_mode():
    for n, path in enumerate(
        tqdm(files[state["count"]:], initial=state["count"],
             total=len(files), desc="Indexing SPLADE"),
        state["count"] + 1
    ):
        paper = json.loads(path.read_text())
        pid, paragraphs = paper["paper_id"], paper["paragraphs"]

        if pid != path.stem:
            raise ValueError(f"Paper ID differs from filename: {path}")

        ids = [None, *paragraphs]
        texts = [
            f'{paper.get("title", "")}. {paper.get("abstract", "")}',
            *paragraphs.values()
        ]
        paper_vector = np.zeros(dimension, dtype=np.float32)

        for i in range(0, len(texts), GROUP):
            vectors = encode(
                model, texts[i:i + GROUP], pooling="max",
                batch_size=args.batch_size, max_tokens=TOKENS
            ).float().cpu().numpy()

            np.maximum(paper_vector, vectors.max(axis=0), out=paper_vector)
            for docid, vector in zip(ids[i:i + GROUP], vectors):
                if docid is not None:
                    sharding.add(buffers, "paragraphs", docid, vector)

        sharding.add(buffers, "papers", pid, paper_vector)

        if n % args.interval == 0 or n == len(files):
            sharding.save(OUT, buffers, dimension, state, n, path.name)
            buffers = sharding.init()
            tqdm.write(f"Saved through {path.name} ({n}/{len(files)})")