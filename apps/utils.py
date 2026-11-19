"""File paths used by more than one module."""

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
