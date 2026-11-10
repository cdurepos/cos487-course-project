"""Steady-state query latency for every neural system timed on Study.

Reads the per-query latency_ms maps in
data/evaluation/{dense,sparse}/study_{paper,paragraph}.json.

The first Study query is dropped. That call loads the model or index, and
for three paragraph runs it is one to two orders of magnitude slower than
the queries that follow. BM25 has retrieval metrics and no timed queries
in these files, so it is omitted.

Bar order matches the graded nDCG order used in metrics-bar.py, best at
the top, so the two figures can be read side by side. Color marks family.

Writes apps/figures/output/latency.pdf and latency.png.

Run from the repository root:

    python apps/figures/latency-bar.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
EVAL = ROOT / "data" / "evaluation"

SPLADE = "#0072B2"
QWEN = "#D55E00"
BGE = "#009E73"

PAPER = (
    ("qwen-finetuned-from-paragraphs", "Qwen fine-tuned, best paragraph", QWEN),
    ("bge-pretrained-paper", "BGE pretrained reranker", BGE),
    ("qwen-pretrained-from-paragraphs", "Qwen pretrained, best paragraph", QWEN),
    ("bge-finetuned-paper", "BGE fine-tuned reranker", BGE),
    ("splade-finetuned", "SPLADE fine-tuned, MaxP", SPLADE),
    ("splade-baseline", "SPLADE baseline, MaxP", SPLADE),
    ("qwen-pretrained-mean-pooling", "Qwen pretrained, mean pool", QWEN),
    ("qwen-finetuned-mean-pooling", "Qwen fine-tuned, mean pool", QWEN),
    ("splade-baseline-pool", "SPLADE baseline, pooled", SPLADE),
    ("splade-finetuned-pool", "SPLADE fine-tuned, pooled", SPLADE),
)

PARAGRAPH = (
    ("bge-finetuned-paragraph", "BGE fine-tuned reranker", BGE),
    ("bge-pretrained-paragraph", "BGE pretrained reranker", BGE),
    ("qwen-finetuned-cosine-similarity", "Qwen fine-tuned, cosine", QWEN),
    ("qwen-pretrained-cosine-similarity", "Qwen pretrained, cosine", QWEN),
    ("splade-finetuned", "SPLADE fine-tuned", SPLADE),
    ("splade-baseline", "SPLADE baseline", SPLADE),
)


def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def steady_means(level):
    means = {}
    for family in ("dense", "sparse"):
        summary = json.loads((EVAL / family / f"study_{level}.json").read_text())
        for name, per_query in summary["performance"]["latency_ms"].items():
            ordered = [value for _, value in sorted(per_query.items(), key=lambda item: int(item[0]))]
            kept = ordered[1:]
            means[name] = sum(kept) / len(kept)
    return means


def draw(ax, models, means, title):
    labels = [label for _, label, _ in models]
    colors = [color for _, _, color in models]
    values = [means[key] for key, _, _ in models]
    y = list(range(len(models)))
    ax.barh(y, values, color=colors, height=0.72)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("Mean query latency (ms)")
    peak = max(values)
    ax.set_xlim(0, peak * 1.18)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)
    for pos, value in zip(y, values):
        ax.text(value + peak * 0.015, pos, f"{value:.0f}", va="center", fontsize=8, color="#222222")


def main():
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    draw(axes[0], PAPER, steady_means("paper"), "Paper")
    draw(axes[1], PARAGRAPH, steady_means("paragraph"), "Paragraph")
    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "latency.pdf", bbox_inches="tight")
    fig.savefig(OUT / "latency.png", dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT / 'latency.pdf'}")
    print(f"Wrote {OUT / 'latency.png'}")


if __name__ == "__main__":
    main()
