
"""Resumable sparse indexes saved as SciPy .npz batches."""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse

LEVELS = ("paragraphs", "papers")


def init():
    return {level: {"ids": [], "vectors": []} for level in LEVELS}


def add(buffers, level, docid, vector):
    buffers[level]["ids"].append(docid)
    buffers[level]["vectors"].append(
        sparse.csr_matrix(vector.reshape(1, -1))
    )


def resume(folder, files, signature, interval):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "checkpoint.json"

    if path.exists():
        state = json.loads(path.read_text())
    else:
        if any(folder.glob("*.npz")):
            raise RuntimeError("Found indexes without a checkpoint")
        state = dict(
            count=0, last_paper=None,
            signature=signature, interval=interval
        )
        path.write_text(json.dumps(state))

    n = state["count"]
    if (
        state["signature"] != signature
        or state["interval"] != interval
        or n > len(files)
        or (n and files[n - 1].name != state["last_paper"])
    ):
        raise RuntimeError("Checkpoint does not match corpus or model")

    for i in range(1, (n + interval - 1) // interval + 1):
        for level in LEVELS:
            name = f"{level}_{i:04d}"
            if (
                not (folder / f"{name}.npz").exists()
                or not (folder / f"{name}_ids.json").exists()
            ):
                raise RuntimeError(f"Missing committed index: {name}")

    return state


def save(folder, buffers, dimension, state, count, last_paper):
    folder = Path(folder)
    shard = (count - 1) // state["interval"] + 1

    for level, b in buffers.items():
        matrix = (
            sparse.vstack(b["vectors"], format="csr")
            if b["vectors"]
            else sparse.csr_matrix((0, dimension))
        )

        name = folder / f"{level}_{shard:04d}"
        temp = name.with_suffix(".tmp.npz")
        sparse.save_npz(temp, matrix)
        temp.replace(name.with_suffix(".npz"))

        temp = folder / f"{level}_{shard:04d}_ids.tmp.json"
        temp.write_text(json.dumps(b["ids"]))
        temp.replace(folder / f"{level}_{shard:04d}_ids.json")

    next_state = {
        **state,
        "count": count,
        "last_paper": last_paper
    }

    temp = folder / "checkpoint.tmp.json"
    temp.write_text(json.dumps(next_state))
    temp.replace(folder / "checkpoint.json")
    state.update(next_state)


def load(folder, level):
    folder = Path(folder)
    state = json.loads(
        (folder / "checkpoint.json").read_text()
    )

    count = (
        state["count"] + state["interval"] - 1
    ) // state["interval"]

    for i in range(1, count + 1):
        name = f"{level}_{i:04d}"

        ids = json.loads(
            (folder / f"{name}_ids.json").read_text()
        )
        matrix = sparse.load_npz(
            folder / f"{name}.npz"
        )

        if len(ids) != matrix.shape[0]:
            raise ValueError(f"Index/ID mismatch: {name}")

        yield ids, matrix


def search(folder, level, query_vectors, top_k=100,
           strategy="maxp", timed=False, normalized=True):
    """Search paragraphs or papers using MaxP or pooled embeddings."""
    if level not in ("paper", "paragraph"):
        raise ValueError("level must be 'paper' or 'paragraph'")

    if level == "paper" and strategy not in ("maxp", "pool"):
        raise ValueError("strategy must be 'maxp' or 'pool'")

    if top_k < 1:
        raise ValueError("top_k must be positive")

    maxp = level == "paper" and strategy == "maxp"
    source = (
        "papers"
        if level == "paper" and strategy == "pool"
        else "paragraphs"
    )

    queries = np.asarray(query_vectors)
    single = queries.ndim == 1

    if single:
        queries = queries[None, :]

    results = [{} for _ in queries]
    times = [0.0] * len(queries)

    for ids, matrix in load(folder, source):
        if not ids:
            continue

        # Document L2 normalization
        if normalized:
            norms = np.sqrt(
                np.asarray(
                    matrix.multiply(matrix).sum(axis=1)
                ).ravel()
            ).clip(min=1e-12)

        # Group paragraphs by their parent paper.
        if maxp:
            paper_ids, groups = np.unique(
                [docid.rsplit("_", 1)[0] for docid in ids],
                return_inverse=True
            )

        for j, query in enumerate(queries):
            start = perf_counter()

            scores = np.asarray(matrix @ query).ravel()

            if normalized:
                qnorm = max(np.linalg.norm(query), 1e-12)
                scores /= norms

            if maxp:
                # Highest paragraph score for each paper.
                values = np.full(len(paper_ids), -np.inf)
                np.maximum.at(values, groups, scores)
                docids = paper_ids
            else:
                values, docids = scores, ids

            # Keep the top candidates from this shard.
            k = min(top_k, len(docids))
            top = np.argpartition(values, -k)[-k:]

            # Merge candidates across shards.
            for i in top:
                docid = str(docids[i])
                score = float(values[i])

                results[j][docid] = max(
                    results[j].get(docid, -np.inf),
                    score
                )

            results[j] = dict(sorted(
                results[j].items(),
                key=lambda item: (-item[1], item[0])
            )[:top_k])

            times[j] += (perf_counter() - start) * 1000

    ranked = [
        sorted(
            hits.items(),
            key=lambda item: (-item[1], item[0])
        )
        for hits in results
    ]

    output = ranked[0] if single else ranked

    if timed:
        return output, times[0] if single else times

    return output