
"""Paragraph-only, aggregation-aware SPLADE fine-tuning."""
import json
import random
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sentence_transformers import SparseEncoder

ROOT = Path(__file__).resolve().parents[2]
DATA, OUT = ROOT / "data", ROOT / "models" / "splade"
TOKENS, EPOCHS, BATCH = 256, 3, 8

random.seed(42)
torch.manual_seed(42)

model = SparseEncoder("naver/splade-v3")
model.max_seq_length = TOKENS
model.gradient_checkpointing_enable()
tok, device = model.tokenizer, model.device
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)


@lru_cache(maxsize=512)
def paragraph(docid):
    pid = docid.rsplit("_", 1)[0]
    return json.loads(
        (DATA / "JSON Files" / f"{pid}.json").read_text()
    )["paragraphs"][docid]


@lru_cache(maxsize=8192)
def chunks(text):
    ids = tok(text, add_special_tokens=False, verbose=False)["input_ids"]
    length = TOKENS - 2
    return [
        [tok.cls_token_id, *ids[i:i + length], tok.sep_token_id]
        for i in range(0, len(ids), length)
    ] or [[tok.cls_token_id, tok.sep_token_id]]


def embed(text):
    parts = chunks(text)
    pooled = []
    for i in range(0, len(parts), BATCH):
        inputs = tok.pad(
            {"input_ids": parts[i:i + BATCH]},
            return_tensors="pt"
        ).to(device)
        pooled.append(model(inputs)["sentence_embedding"].amax(dim=0))
    return torch.stack(pooled).amax(dim=0)


# Load training queries
queries = json.loads((DATA / "Train.json").read_text())
queries = queries.get("queries", queries) if isinstance(queries, dict) else queries

if isinstance(queries, list):
    queries = {
        str(q.get("qid", q.get("query_id", q.get("id")))):
        q.get("query", q.get("text", q.get("question")))
        for q in queries
    }
else:
    queries = {
        str(k): v if isinstance(v, str) else v.get(
            "query", v.get("text", v.get("question"))
        )
        for k, v in queries.items()
    }

# Load paragraph QRELs; prefer relevance-2 positives
qrels = {}
for line in (DATA / "qrels" / "Train_paragraph_qrel.tsv").read_text().splitlines():
    qid, _, docid, rel = line.split()
    if qid in queries and queries[qid] and int(rel) > 0:
        qrels.setdefault(qid, []).append((int(rel), docid))

positives = {
    qid: [docid for rel, docid in matches
          if rel == max(r for r, _ in matches)]
    for qid, matches in qrels.items()
}

# Query-disjoint training and validation
qids = sorted(positives)
assert len(qids) >= 4, "Need at least four queries with relevant paragraphs"
valid = set(random.Random(42).sample(qids, max(2, len(qids) // 5)))

OUT.mkdir(parents=True, exist_ok=True)
history = []

# Fine-tuning
for epoch in range(EPOCHS):
    results = []

    for training in (True, False):
        model.train(training)
        ids = [qid for qid in qids if (qid not in valid) == training]
        if training:
            random.shuffle(ids)

        losses = []
        with torch.set_grad_enabled(training):
            for i in range(0, len(ids) - 1, BATCH):
                batch = ids[i:i + BATCH]
                docs = [
                    random.choice(positives[qid]) if training
                    else positives[qid][0]
                    for qid in batch
                ]
                if len(set(docs)) != BATCH:
                    continue

                q = torch.stack([embed(queries[qid]) for qid in batch])
                d = torch.stack([embed(paragraph(docid)) for docid in docs])

                loss = F.cross_entropy(
                    q @ d.T / 20, torch.arange(BATCH, device=device)
                )
                loss += (
                    5e-5 * q.mean(0).square().sum()
                    + 3e-5 * d.mean(0).square().sum()
                )

                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                losses.append(loss.item())

        assert losses, "No usable batches: check QRELs or duplicate positives"
        results.append(sum(losses) / len(losses))

    history.append({
        "epoch": epoch + 1,
        "train_loss": results[0],
        "val_loss": results[1]
    })
    print(history[-1])

    if results[1] == min(h["val_loss"] for h in history):
        model.save_pretrained(str(OUT))

# Save training history and loss curves
(OUT / "loss_history.json").write_text(json.dumps(history, indent=2))

for key in ("train_loss", "val_loss"):
    plt.plot(
        [h["epoch"] for h in history],
        [h[key] for h in history],
        marker="o", label=key
    )

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(OUT / "loss_curve.png", bbox_inches="tight")
plt.close()