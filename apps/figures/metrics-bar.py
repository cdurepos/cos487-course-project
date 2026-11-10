"""Study-set metric comparison for every scored system, including BM25.

Reads data/evaluation/dense/study_{paper,paragraph}.json and the matching
sparse summaries. BM25 is identical in both files and is drawn once.

Graded bars use relevance level 1 (relevant or partially relevant):
nDCG, nDCG@5, MRR, P@5, P@10.
Strict bars use relevance level 2 (relevant only): MRR and P@5.

Systems are ordered by graded nDCG, highest at the top. Bar color marks
the family so the same system can be tracked across panels.

Writes:
    apps/figures/output/metrics-paper.pdf
    apps/figures/output/metrics-paragraph.pdf
    apps/figures/output/metrics-strict.pdf
and PNG copies of each.

Run from the repository root:

    python apps/figures/metrics-bar.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
EVAL = ROOT / "data" / "evaluation"

# Graded nDCG order, best first. Color is the model family.
BM25 = "#333333"
SPLADE = "#0072B2"
QWEN = "#D55E00"
BGE = "#009E73"

PAPER = (
    ("qwen-finetuned-from-paragraphs", "Qwen fine-tuned\nbest paragraph", QWEN),
    ("bge-pretrained-paper", "BGE pretrained\nreranker", BGE),
    ("qwen-pretrained-from-paragraphs", "Qwen pretrained\nbest paragraph", QWEN),
    ("bge-finetuned-paper", "BGE fine-tuned\nreranker", BGE),
    ("splade-finetuned", "SPLADE fine-tuned\nMaxP", SPLADE),
    ("bm25", "BM25", BM25),
    ("splade-baseline", "SPLADE baseline\nMaxP", SPLADE),
    ("qwen-pretrained-mean-pooling", "Qwen pretrained\nmean pool", QWEN),
    ("qwen-finetuned-mean-pooling", "Qwen fine-tuned\nmean pool", QWEN),
    ("splade-baseline-pool", "SPLADE baseline\npooled", SPLADE),
    ("splade-finetuned-pool", "SPLADE fine-tuned\npooled", SPLADE),
)

PARAGRAPH = (
    ("bge-finetuned-paragraph", "BGE fine-tuned\nreranker", BGE),
    ("bge-pretrained-paragraph", "BGE pretrained\nreranker", BGE),
    ("qwen-finetuned-cosine-similarity", "Qwen fine-tuned\ncosine", QWEN),
    ("qwen-pretrained-cosine-similarity", "Qwen pretrained\ncosine", QWEN),
    ("bm25", "BM25", BM25),
    ("splade-finetuned", "SPLADE fine-tuned", SPLADE),
    ("splade-baseline", "SPLADE baseline", SPLADE),
)

GRADED = (
    ("ndcg", "nDCG"),
    ("ndcg@5", "nDCG@5"),
    ("mrr", "MRR"),
    ("precision@5", "P@5"),
    ("precision@10", "P@10"),
)
STRICT = (
    ("mrr", "MRR"),
    ("precision@5", "P@5"),
)


def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def load_level(level):
    merged = {}
    for family in ("dense", "sparse"):
        path = EVAL / family / f"study_{level}.json"
        summary = json.loads(path.read_text())
        for pass_name, block in summary["relevance_passes"].items():
            merged.setdefault(pass_name, {}).update(block["metrics"])
    return merged


def draw(axes, models, metrics, scores):
    labels = [label for _, label, _ in models]
    colors = [color for _, _, color in models]
    y = list(range(len(models)))
    for ax, (key, title) in zip(axes, metrics):
        values = [scores[key_name][key] for key_name, _, _ in models]
        ax.barh(y, values, color=colors, height=0.72)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Score")
        ax.set_xlim(0, max(values) * 1.22)
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
        ax.set_axisbelow(True)
        for pos, value in zip(y, values):
            ax.text(value + max(values) * 0.02, pos, f"{value:.3f}", va="center", fontsize=7.5, color="#222222")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT / name}.pdf")


def graded_figure(level, models, height):
    scores = load_level(level)["relevant_or_partial"]
    fig, axes = plt.subplots(1, len(GRADED), figsize=(14.6, height), sharey=True)
    draw(axes, models, GRADED, scores)
    fig.tight_layout()
    save(fig, f"metrics-{level}")
    plt.close(fig)


def strict_figure(paper_models, paragraph_models):
    fig, axes = plt.subplots(2, len(STRICT), figsize=(7.4, 8.2), sharex="col")
    panels = (
        (0, "Paper", paper_models),
        (1, "Paragraph", paragraph_models),
    )
    for row, level_name, models in panels:
        scores = load_level(level_name.lower())["relevant_only"]
        draw(axes[row], models, STRICT, scores)
        axes[row, 0].set_ylabel(level_name)
    fig.tight_layout()
    save(fig, "metrics-strict")
    plt.close(fig)


def main():
    apply_style()
    graded_figure("paper", PAPER, 5.6)
    graded_figure("paragraph", PARAGRAPH, 4.2)
    strict_figure(PAPER, PARAGRAPH)


if __name__ == "__main__":
    main()
