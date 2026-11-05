
"""Index arXiv papers and paragraphs with resumable SPLADE CSR shards."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SparseEncoder
from tqdm.auto import tqdm

from apps.retrieval.sparse import catalog
from apps.utils.encoding import encode, get_output_dim
from apps.utils import sharding

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "JSON Files"

TOKENS, GROUP = 256, 16

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--interval", type=int, default=100)
args = parser.parse_args()

if args.batch_size < 1 or args.interval < 1:
    parser.error("Invalid batch size or interval")

for item in catalog().values():
    MODEL, OUT = item["checkpoint"], item["index"]
    files = sorted(DATA.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No paper JSON files found in {DATA}")

    if isinstance(MODEL, Path):
        model_spec = [
            [str(p.relative_to(MODEL)), p.stat().st_size, p.stat().st_mtime_ns]
            for p in sorted(MODEL.rglob("*")) if p.is_file()
            and p.suffix in (".bin", ".safetensors", ".json", ".txt", ".model")
            and p.name != "loss_history.json"
        ]
    else:
        model_spec = MODEL

    spec = {
        "corpus": [[p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in files],
        "model": model_spec,
        "tokens": TOKENS, "interval": args.interval,
        "abstract": True,
    }
    signature = hashlib.sha256(json.dumps(spec).encode()).hexdigest()
    state = sharding.resume(OUT, files, signature, args.interval)

    model = SparseEncoder(str(MODEL))
    model.max_seq_length = TOKENS
    model.eval()
    dimension = get_output_dim(model)
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

            ids = [f"{pid}_0", *paragraphs]
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
                    sharding.add(buffers, "paragraphs", docid, vector)

            sharding.add(buffers, "papers", pid, paper_vector)

            if n % args.interval == 0 or n == len(files):
                sharding.save(OUT, buffers, dimension, state, n, path.name)
                buffers = sharding.init()
                tqdm.write(f"Saved through {path.name} ({n}/{len(files)})")