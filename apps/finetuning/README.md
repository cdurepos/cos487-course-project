# Fine-tuning

Fine-tunes SPLADE-v3 on the Train queries and paragraph relevance judgments. Indexing and retrieval use the saved encoder.

## Run

From the repository root:

```bash
python -m apps.finetuning.sparse
```

Reads `data/Train.json` and `data/qrels/Train_paragraph_qrel.tsv`. Writes the checkpoint with the lowest validation loss to the `splade-finetuned` directory in `configs/sparse.yaml`, plus `loss_history.json` and `loss_curve.png`.

Training uses paragraph text only. About 80% of the eligible queries are used for training. The rest form a query-disjoint validation set. Study and Test queries are excluded.
