"""File paths and intent prompts used by more than one module."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

CORPUS_DIR = DATA_DIR / "JSON Files"
QRELS_DIR = DATA_DIR / "qrels"
EVAL_DIR = DATA_DIR / "evaluation"

MODEL_DIR = ROOT_DIR / "models" / "finetuned_qwen"

INDEX_DIR = DATA_DIR / "indexes"
INDEX_MAP = {
    "pretrained_paragraph": INDEX_DIR / "qwen_pretrained.npz",
    "pretrained_paper": INDEX_DIR / "qwen_pretrained_papers.npz",
    "finetuned_paragraph": INDEX_DIR / "qwen_finetuned.npz",
    "finetuned_paper": INDEX_DIR / "qwen_finetuned_papers.npz",
}

PROMPT_INTENT_DOMINANT = """
Read the question and identify its single dominant intent.
Write one well-suited retrieval query for that intent. Keep the scientific terms that identify the topic.
Do not answer the question. Return only the query, on one line.

Question: {query}
"""

PROMPT_INTENT_MULTI = """
Read the research question and identify its {k} most likely intents, most likely first.
Write one well-suited retrieval query for each intent. The queries must target different information needs.
Keep the scientific terms that identify the topic. Do not answer the question.
Return exactly {k} lines, one query per line, with no numbering or explanation.

Question: {query}
"""
