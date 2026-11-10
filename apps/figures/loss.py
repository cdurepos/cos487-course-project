"""Training loss against validation loss for the three fine-tuned models.

Reads:
    data/dense-train-hist/training_history_bge_crossencoder.json
    data/dense-train-hist/training_history_qwen_biencoder.json
    models/splade-finetuned/loss_history.json

Writes apps/figures/output/loss.pdf and loss.png.
The marked epoch is the one with the lowest validation loss, which is the
checkpoint each training script saves.

Run from the repository root:

    python apps/figures/loss.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"

TRAIN = "#0072B2"
VAL = "#D55E00"

SERIES = (
    {
        "title": "BGE reranker",
        "path": ROOT / "data/dense-train-hist/training_history_bge_crossencoder.json",
        "val_key": "validation_loss",
    },
    {
        "title": "Qwen3 bi-encoder",
        "path": ROOT / "data/dense-train-hist/training_history_qwen_biencoder.json",
        "val_key": "validation_loss",
    },
    {
        "title": "SPLADE",
        "path": ROOT / "models/splade-finetuned/loss_history.json",
        "val_key": "val_loss",
    },
)


def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def load(spec):
    rows = json.loads(spec["path"].read_text())
    epochs = [row["epoch"] for row in rows]
    train = [row["train_loss"] for row in rows]
    val = [row[spec["val_key"]] for row in rows]
    best = min(range(len(val)), key=val.__getitem__)
    return epochs, train, val, best


def main():
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7))

    for ax, spec in zip(axes, SERIES):
        epochs, train, val, best = load(spec)
        ax.plot(epochs, train, color=TRAIN, marker="o", markersize=4.5, linewidth=1.6, label="Training loss")
        ax.plot(epochs, val, color=VAL, marker="o", markersize=4.5, linewidth=1.6, label="Validation loss")
        ax.scatter(
            [epochs[best]], [val[best]],
            s=70, facecolors="none", edgecolors=VAL, linewidths=1.6, zorder=4,
            label="Lowest validation loss",
        )
        ax.set_title(spec["title"])
        ax.set_xlabel("Epoch")
        ax.set_xticks(epochs)
        ax.set_xlim(min(epochs) - 0.3, max(epochs) + 0.3)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Loss")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "loss.pdf", bbox_inches="tight")
    fig.savefig(OUT / "loss.png", dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT / 'loss.pdf'}")
    print(f"Wrote {OUT / 'loss.png'}")


if __name__ == "__main__":
    main()
