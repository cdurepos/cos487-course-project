"""BM25 search for an original question or a saved model reply.

    retrieve(question)
    retrieve(reply, "dominant")
    retrieve(reply, "multi")
    retrieve(reply, "likelihood", original=question)

"multi" reads one query per line. "likelihood" reads a likelihood, a tab,
and a query on each line. Any other approach searches the whole reply as one query.
"""

from apps.baseline.bm25 import retrieve as bm25_retrieve


def retrieve(query, approach=None, level="paragraph", k=100, original=None):
    """Return the top-k (doc id, score) pairs for one research question."""
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

    # Stemmed BM25, at the paragraph or paper index for this level.
    rankings = [
        list(bm25_retrieve(line, level, True, k).items())
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
