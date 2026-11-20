"""Query, then an approach prompt, then Qwen retrieval.

    retrieve(question, "dominant")
    retrieve(question, "multi", i=3)
    retrieve(question, "likelihood", i=3)

"multi" reads one query per line. "likelihood" reads a likelihood, a tab,
and a query on each line. Any other approach searches the whole reply as one query.
"""

from apps.dense.retrieval import search
from apps.gemma import generate, load_model
from apps.utils import PROMPT_MAP

_model = None
_assistant = None
_processor = None

def retrieve(query, approach=None, training="finetuned", level="paragraph", k=100, i=3):
    """Return the top-k (doc id, score) pairs for one research question."""
    global _model, _assistant, _processor
    # Load Gemma once and reuse it for every question.
    if _model is None:
        _model, _assistant, _processor = load_model()

    if approach is None:
        text = query
    else:
        try:
            text = generate(
                messages=[{"role": "user", "content": PROMPT_MAP[approach].format(query=query, i=i)}],
                model=_model,
                processor=_processor,
                assistant=_assistant,
                max_new_tokens=512,
            )
        except ValueError as e:
            print(f"Approach \"{approach}\" not yet implemented. Available approaches are: {[str(val) for val in PROMPT_MAP.keys()]}")

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
        parsed = [(1.0, query)]

    # Paper search always uses the paper's best paragraph.
    rankings = [
        search(line, training=training, level=level, paper_method="from_paragraphs", k=k)
        for _, line in parsed
    ]
    # One query keeps the Qwen scores.
    if len(rankings) == 1:
        return rankings[0]

    # Several queries: reciprocal rank fusion. Multi-intent weights are all 1.
    fused = {}
    for (weight, _), ranking in zip(parsed, rankings):
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (60 + rank)
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)[:k]
