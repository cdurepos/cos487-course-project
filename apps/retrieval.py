"""Project search: load an index and the BM25 parameters, then score a reply.

    retrieve(question)
    retrieve(reply, "dominant")
    retrieve(reply, "multi")
    retrieve(reply, "likelihood", original=question)

"multi" reads one query per line. "likelihood" reads a likelihood, a tab,
and a query on each line. Any other approach searches the whole reply as one query.
"""

import yaml

from apps.baseline.bm25 import retrieve as bm25_retrieve
from apps.baseline.index import load_index
from apps.utils import CONFIG_PATH, INDEX_MAP

CONFIG = yaml.safe_load(CONFIG_PATH.read_text()) or {}
INDEXES = {}

def load_param(level, approach=None, k_1=None, b=None):
    """k1 and b for a level, using the approach block when grid search saved one.

    Passed-in k_1 and b replace the config values.
    """
    level_params = CONFIG.get(level) or {}
    chosen = level_params
    if approach:
        approach_params = level_params.get(approach)
        if isinstance(approach_params, dict):
            chosen = approach_params
    default = CONFIG.get("default") or {}
    if k_1 is None:
        k_1 = chosen.get("k_1", level_params.get("k_1", default.get("k_1", 1.2)))
    if b is None:
        b = chosen.get("b", level_params.get("b", default.get("b", 0.75)))
    return float(k_1), float(b)


def retrieve(query, approach=None, level="paragraph", k=100, original=None, k_1=None, b=None):
    """Return the top-k (doc id, score) pairs for one research question.

    k_1 and b override the parameters in config.yaml. Leave them unset to use
    the saved values for this level and approach.
    """
    fallback = original if original is not None else query
    text = query
    if approach is None:
        parsed = [(1.0, text)]
    else:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if approach == "likelihood":
            # One line is "0.55\tquery". 55 and 55% count as 0.55.
            parsed = []
            for line in lines:
                weight, sep, line = line.partition("\t")
                if not sep:
                    continue
                try:
                    value = float(weight.strip().rstrip("%"))
                except ValueError:
                    continue
                if value > 1:
                    value /= 100
                if 0 <= value <= 1 and line.strip():
                    parsed.append((value, line.strip()))
            total = sum(weight for weight, _ in parsed)
            if total > 0:
                parsed = [(weight / total, line) for weight, line in parsed]
        elif approach == "multi":
            parsed = [(1.0, line) for line in lines]
        else:
            # One passage: rewrite, dominant intent, or a generated answer.
            parsed = [(1.0, " ".join(text.split()))]
        if not parsed or not parsed[0][1]:
            parsed = [(1.0, fallback)]

    k_1, b = load_param(level, approach, k_1, b)

    # load index & cache
    path = INDEX_MAP[level]
    if path not in INDEXES:
        print(f"Loading index {path.name}", flush=True)
        INDEXES[path] = load_index(path)
    index = INDEXES[path]

    rankings = [
        list(bm25_retrieve(line, index, k, k_1, b).items())
        for _, line in parsed
    ]
    # One query keeps the BM25 scores.
    if len(rankings) == 1:
        return rankings[0]

    # Several queries: reciprocal rank fusion. Multi-intent weights are all 1.
    fused = {}
    for (weight, _), ranking in zip(parsed, rankings):
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (60 + rank)
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)[:k]


def main():
    """Search the project indexes from the command line."""
    while True:
        query = input("Enter a query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        while True:
            level = input("Enter the level of the index to search (paragraph/paper): ").strip().lower()
            if level in INDEX_MAP:
                break
        while True:
            k = input("Enter how many results to return: ").strip()
            try:
                k = int(k)
            except ValueError:
                continue
            if k > 0:
                break
        results = retrieve(query, level=level, k=k)
        print(f"Top {k} results for query '{query}':")
        for doc_id, score in results:
            print(f"{doc_id}: {score:.4f}")
        print("-" * 50)


if __name__ == "__main__":
    main()
