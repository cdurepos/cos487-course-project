"""Query, then an intent prompt, then Qwen retrieval.

Gemma writes the queries. One returned line is searched as written. Several
lines are searched separately and merged with reciprocal rank fusion.

    from apps.utils import INTENT_DOMINANT, INTENT_TOP_K
    retrieve(query, INTENT_DOMINANT)
    retrieve(query, INTENT_TOP_K, intents=3)
"""

from apps.dense.retrieval import search
from apps.gemma import generate, load_model


_model = None
_assistant = None
_processor = None


def retrieve(query, prompt, training="finetuned", level="paragraph", k=100, i=3):
    """Return the top-k (doc id, score) pairs for one research question."""
    global _model, _assistant, _processor
    if _model is None:
        _model, _assistant, _processor = load_model()

    text = generate(
        messages=[{"role": "user", "content": prompt.format(query=query, k=i)}],
        model=_model,
        processor=_processor,
        assistant=_assistant,
        max_new_tokens=256,
    )

    queries = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("-").strip()
        head, _, tail = line.partition(" ")
        if head.rstrip(".)").isdigit() and tail:
            line = tail.strip()
        if line:
            queries.append(line)
    if not queries:
        queries = [query]

    rankings = [
        search(line, training=training, level=level, paper_method="from_paragraphs", k=k)
        for line in queries
    ]
    if len(rankings) == 1:
        return rankings[0]

    fused = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (60 + rank)
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)[:k]
