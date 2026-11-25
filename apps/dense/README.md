# Dense retrieval

Fine-tunes Qwen3-Embedding-0.6B, builds paragraph and paper indexes, and searches them.

## Fine-tuning

From the repository root:

```bash
python -m apps.dense.finetune
```

Reads `data/Train.json` and `data/qrels/Train_paragraph_qrel.tsv`. Writes the checkpoint with the lowest validation loss to `models/qwen/best_adapter`, and the latest checkpoint to `models/qwen/latest`. Training history is `models/qwen/training_history.json`.

About 90% of the eligible queries are used for training. The rest form a query-disjoint validation set.

## Indexing

From the repository root, after fine-tuning:

```bash
python -m apps.dense.indexing
```

Defaults to both the pre-trained and fine-tuned models, and to both paragraph and paper levels. `--batch-size` defaults to 8. `--force` rebuilds an index that already exists.

Paragraph embeddings are saved under `data/indexes`. Paper embeddings are the mean of a paper's paragraph embeddings, its title embedding, and its abstract embedding when the abstract is not already a paragraph.

## Retrieval

Evaluation calls `search()`. There is no separate command.

```python
from apps.dense.retrieval import search, delete_models_and_indexes

search(query, training, level, paper_method, k=100)
delete_models_and_indexes()
```

`search()` encodes the query and searches the specified level. `training` is `pretrained` or `finetuned`. `level` is `paragraph` or `paper`. Paragraph search ranks by cosine similarity. For papers, `paper_method` is `mean_pooling` or `from_paragraphs`. `from_paragraphs` takes each paper's score from its highest-scoring paragraph. The return value is the top-k document ids and scores.
