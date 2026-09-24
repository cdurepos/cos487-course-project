"""GET /search — rank documents with BM25 or TF-IDF."""

from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from apps.processing.preprocess import load_titles
from apps.retrieval import bm25, tfidf

router = APIRouter(tags=["search"])

RetrievalType = Literal["bm25", "tfidf"]
IndexLevel = Literal["paper", "paragraph"]

_titles: dict[str, tuple[str, str]] | None = None


def _get_titles() -> dict[str, tuple[str, str]]:
    global _titles
    if _titles is None:
        _titles = load_titles()
    return _titles


def _paper_id(doc_id: str) -> str:
    """Paragraph IDs look like '2408.00001_8'; paper IDs have no trailing _n."""
    if "_" in doc_id:
        return doc_id.rsplit("_", 1)[0]
    return doc_id


class Hit(BaseModel):
    id: str
    score: float
    title: str = ""
    snippet: str = ""
    url: str = ""


class SearchResponse(BaseModel):
    query: str
    type: RetrievalType
    level: IndexLevel
    stem: bool
    results: list[Hit]
    total_hits: int = Field(serialization_alias="totalHits")


@router.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="Query text."),
    type: RetrievalType = Query(  # noqa: A002
        ...,
        description="Retrieval method: bm25 or tfidf.",
    ),
    level: IndexLevel = Query("paper", description="Index level."),
    stem: bool = Query(True, description="Use the stemmed index."),
    k: int = Query(10, ge=1, le=100, description="Number of results."),
) -> SearchResponse:
    """Return the top-k documents for `q` using the chosen retrieval method."""
    if type == "bm25":
        ranked = bm25.retrieve(q, level, stem, k)
    elif type == "tfidf":
        ranked = tfidf.retrieve(q, level, stem, k)
    else:
        raise HTTPException(status_code=422, detail="type must be bm25 or tfidf")

    titles = _get_titles()
    results: list[Hit] = []
    for doc_id, score in ranked.items():
        paper_id = _paper_id(doc_id)
        title, abstract = titles.get(paper_id, (doc_id, ""))
        results.append(
            Hit(
                id=doc_id,
                score=float(score),
                title=title or doc_id,
                snippet=abstract,
                url=f"https://arxiv.org/abs/{paper_id}",
            )
        )

    return SearchResponse(
        query=q,
        type=type,
        level=level,
        stem=stem,
        results=results,
        total_hits=len(results),
    )
