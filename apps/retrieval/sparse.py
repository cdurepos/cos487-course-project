"""Sparse retrieval over SPLADE paragraph and paper shards."""
import numpy as np
import torch

from apps.utils.encoding import encode
from apps.utils.sharding import load

TOKENS = 256
SHARDS = None


def warmup(index, level, strategy="maxp"):
    """Read this level's shards into memory."""
    global SHARDS
    if level not in ("paper", "paragraph"):
        raise ValueError("level must be 'paper' or 'paragraph'")
    if level == "paper" and strategy not in ("maxp", "pool"):
        raise ValueError("strategy must be 'maxp' or 'pool'")

    source = "papers" if level == "paper" and strategy == "pool" else "paragraphs"
    SHARDS = [
        (ids, matrix)
        for ids, matrix in load(index, source)
        if ids
    ]


def search(model, level, query, top_k=100, strategy="maxp"):
    """Encode queries, score the loaded shards, and cosine-normalize."""
    if SHARDS is None:
        raise RuntimeError("Call warmup() before search()")
    if level not in ("paper", "paragraph"):
        raise ValueError("level must be 'paper' or 'paragraph'")
    if level == "paper" and strategy not in ("maxp", "pool"):
        raise ValueError("strategy must be 'maxp' or 'pool'")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    single = isinstance(query, str)
    pairs = [(None, query)] if single else list(query)
    vectors = []

    with torch.inference_mode():
        for _, text in pairs:
            vector = encode(model, text, pooling="max", max_tokens=TOKENS)
            vectors.append(vector.float().cpu().numpy())

    query_vectors = np.stack(vectors)
    maxp = level == "paper" and strategy == "maxp"
    results = [{} for _ in pairs]

    for ids, matrix in SHARDS:
        norms = np.sqrt(
            np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()
        ).clip(min=1e-12)
        if maxp:
            paper_ids, groups = np.unique(
                [docid.rsplit("_", 1)[0] for docid in ids],
                return_inverse=True,
            )

        for j, vector in enumerate(query_vectors):
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

    ranked = [
        dict(sorted(hits.items(), key=lambda item: (-item[1], item[0])))
        for hits in results
    ]
    if single:
        return ranked[0]
    return {qid: ranked[i] for i, (qid, _) in enumerate(pairs)}
