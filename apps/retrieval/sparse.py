"""Sparse retrieval over SPLADE paragraph and paper shards."""
import statistics
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml

from apps.utils.encoding import encode
from apps.utils.sharding import load

ROOT = Path(__file__).resolve().parents[2]
TOKENS = 256


def catalog():
    """Checkpoint and index for each sparse model in configs/sparse.yaml."""
    listed = yaml.safe_load((ROOT / "configs" / "sparse.yaml").read_text())
    found = {}
    for name, item in listed.items():
        local = ROOT / item["checkpoint"]
        found[name] = {
            "checkpoint": local if local.is_dir() else item["checkpoint"],
            "index": ROOT / item["index"],
            "directory": local,
        }
    return found


def search(model, index, level, query, top_k=100, strategy="maxp", timed=False):
    """Encode queries and rank paragraphs, or papers by MaxP or a pooled vector."""
    if level not in ("paper", "paragraph"):
        raise ValueError("level must be 'paper' or 'paragraph'")
    if level == "paper" and strategy not in ("maxp", "pool"):
        raise ValueError("strategy must be 'maxp' or 'pool'")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    index = Path(index)
    single = isinstance(query, str)
    pairs = [(None, query)] if single else list(query)
    encoding_ms, vectors = [], []

    with torch.inference_mode():
        for _, text in pairs:
            start = perf_counter()
            vector = encode(model, text, pooling="max", max_tokens=TOKENS)
            vectors.append(vector.float().cpu().numpy())
            encoding_ms.append((perf_counter() - start) * 1000)

    query_vectors = np.stack(vectors)
    maxp = level == "paper" and strategy == "maxp"
    source = "papers" if level == "paper" and strategy == "pool" else "paragraphs"
    results = [{} for _ in pairs]
    scoring_ms = [0.0] * len(pairs)
    wall_start = perf_counter()

    for ids, matrix in load(index, source):
        if not ids:
            continue

        norms = np.sqrt(
            np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()
        ).clip(min=1e-12)
        if maxp:
            paper_ids, groups = np.unique(
                [docid.rsplit("_", 1)[0] for docid in ids],
                return_inverse=True,
            )

        for j, vector in enumerate(query_vectors):
            start = perf_counter()
            qnorm = max(float(np.linalg.norm(vector)), 1e-12)
            scores = np.asarray(matrix @ vector).ravel() / (norms * qnorm)
            if level == "paragraph":
                scores[[i for i, docid in enumerate(ids) if str(docid).endswith("_0")]] = -np.inf
            if maxp:
                values = np.full(len(paper_ids), -np.inf)
                np.maximum.at(values, groups, scores)
                docids = paper_ids
            else:
                values, docids = scores, ids

            k = min(top_k, len(docids))
            for i in np.argpartition(values, -k)[-k:]:
                docid, score = str(docids[i]), float(values[i])
                results[j][docid] = max(results[j].get(docid, -np.inf), score)
            results[j] = dict(sorted(
                results[j].items(), key=lambda item: (-item[1], item[0]),
            )[:top_k])
            scoring_ms[j] += (perf_counter() - start) * 1000

    wall_ms = (perf_counter() - wall_start) * 1000
    ranked = [
        dict(sorted(hits.items(), key=lambda item: (-item[1], item[0])))
        for hits in results
    ]
    output = ranked[0] if single else {
        qid: ranked[i] for i, (qid, _) in enumerate(pairs)
    }
    if not timed:
        return output

    durations = [a + b for a, b in zip(encoding_ms, scoring_ms)][1:]
    files = [*index.glob(f"{source}_*.npz"), *index.glob(f"{source}_*_ids.json")]
    return output, {
        "queries": len(durations), "warmup_excluded": 1,
        "mean": round(statistics.mean(durations), 3) if durations else None,
        "median": round(statistics.median(durations), 3) if durations else None,
        "measurement": "query encoding + scoring (excludes shard I/O)",
        "shard_search_wall_ms": round(wall_ms, 3),
    }, {
        "path": str(index.relative_to(ROOT)) if index.is_relative_to(ROOT) else str(index),
        "source": source,
        "strategy": strategy if level == "paper" else "paragraph",
        "bytes": sum(p.stat().st_size for p in files),
        "files": len(files),
    }
