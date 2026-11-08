"""
Creates loss diagrams with the training and validation loss (per epoch) 
for fine-tuning the qlora adapter for the qwen model, and for fine-tuning the bge model

Writes to loss_diagram.pdf in each model's outputs directory

Usage:
    python -m apps.finetuning.create_dense_loss_diagram
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
from pathlib import Path

models_dir = Path(__file__).parent.parent.parent / "models"

for model in ["qwen", "bge_reranker"]:
    finetuning_outputs_dir = models_dir / model
    training_history_file = finetuning_outputs_dir / "training_history.json"
    output_diagram_file = finetuning_outputs_dir / "loss_diagram.pdf"

    with open(training_history_file, "r") as file:
        training_history = json.load(file)
    losses = [(epoch["epoch"], epoch["train_loss"], epoch["validation_loss"]) for epoch in training_history]

    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure(figsize=(10, 6))
    plt.plot([epoch[0] for epoch in losses], [epoch[1] for epoch in losses], label="Training Loss", color="purple", marker="o")
    plt.plot([epoch[0] for epoch in losses], [epoch[2] for epoch in losses], label="Validation Loss", color="blue", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.savefig(output_diagram_file)
