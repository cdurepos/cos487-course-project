"""
Fine-tune BGE reranker using FlagEmbedding.

Usage:
    python -m apps.finetuning.dense_crossencoder
"""


from __future__ import annotations

import gc
import json
import random
import re
import shutil
import subprocess
import sys
import os

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from tqdm import tqdm

from FlagEmbedding import FlagReranker


ROOT = Path(__file__).resolve().parent.parent.parent

TRAIN_TOPICS = ROOT / "data" / "Train.json"
PARAGRAPH_QRELS = ROOT / "data" / "qrels" / "Train_paragraph_qrel.tsv"
PARAGRAPH_CORPUS = ROOT / "data" / "JSON Files"

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
OUTPUT_DIR = ROOT / "models" / "bge_reranker"
TRAINING_DIR = OUTPUT_DIR / "training_data"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
BEST_MODEL_DIR = OUTPUT_DIR / "best_model"
CACHE_DIR = OUTPUT_DIR / "cache"

MODEL_MAX_LENGTH = 512
QUERY_MAX_LENGTH = 256
PASSAGE_MAX_LENGTH = 512

TRAIN_EPOCHS = 10
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 6e-5
WEIGHT_DECAY = 0.01
TRAIN_GROUP_SIZE = 8

VALIDATION_SIZE = 0.1
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_corpus(path: Path) -> dict[str, str]:
    """Loads each paper's paragraphs"""

    if not path.exists():
        raise FileNotFoundError(f"Paragraph corpus not found: {path}")

    paragraphs = {}
    for file in tqdm(sorted(os.listdir(path)), desc="Loading corpus"):
        if not file.endswith(".json"):
            continue

        with (path / file).open("r", encoding="utf-8") as f:
            data = json.load(f)
        if "paragraphs" not in data:
            print(f"Warning: No 'paragraphs' key found in {file}. Skipping.")
            continue

        for paragraph_id, paragraph_text in data["paragraphs"].items():
            text = str(paragraph_text).strip()
            if text:
                paragraphs[str(paragraph_id)] = text
    if not paragraphs:
        raise ValueError(f"No paragraphs were loaded from {path}")
    return paragraphs


def split_training_queries(
    topics: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]],
    corpus: Mapping[str, str],
    validation_size: float,
    seed: int,
):
    """Split queries that have at least one judment for each grade"""

    valid_qids = []
    for qid in topics:
        judgments = qrels.get(qid, {})
        has_positive = any(grade >= 2 and docid in corpus for docid, grade in judgments.items())
        has_negative = any(grade == 1 and docid in corpus for docid, grade in judgments.items())
        if has_positive and has_negative:
            valid_qids.append(qid)

    valid_qids.sort()
    rng = random.Random(seed)
    rng.shuffle(valid_qids)

    n_validation = max(1, int(round(len(valid_qids) * validation_size)))
    n_validation = min(n_validation, len(valid_qids) - 1)

    validation_qids = set(valid_qids[:n_validation])
    train_qids = set(valid_qids[n_validation:])

    print(f"Valid queries: {len(valid_qids):,}")
    print(f"Training queries: {len(train_qids):,}")
    print(f"Validation queries: {len(validation_qids):,}")
    return train_qids, validation_qids


def build_training_jsonl(
    topics: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]],
    corpus: Mapping[str, str],
    included_qids: set[str],
    output_path: Path,
):
    """Create one training row per valid query (with at least one paragraph for each grade)"""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    positive_count = 0
    negative_count = 0
    skipped_count = 0

    with open(output_path, "w") as file:
        for qid in sorted(included_qids):
            query = topics.get(qid, "").strip()
            if not query:
                skipped_count += 1
                continue

            judgments = qrels.get(qid, {})
            positives = sorted({
                docid
                for docid, grade in judgments.items()
                if grade >= 2
                and docid in corpus
                and corpus[docid].strip()
            })
            negatives = sorted({
                docid
                for docid, grade in judgments.items()
                if grade == 1
                and docid in corpus
                and corpus[docid].strip()
            })

            if not positives or not negatives:
                skipped_count += 1
                continue
            positive_texts = [corpus[docid].strip() for docid in positives]
            negative_texts = [corpus[docid].strip() for docid in negatives]

            row = {
                "query": query,
                "pos": positive_texts,
                "neg": negative_texts,
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            row_count += 1
            positive_count += len(positive_texts)
            negative_count += len(negative_texts)
    if row_count == 0:
        raise RuntimeError("No usable training rows were created.")

    print("\nTraining JSONL statistics")
    print(f"  Queries/rows: {row_count:,}")
    print(f"  Positive passages: {positive_count:,}")
    print(f"  Negative passages: {negative_count:,}")
    print(f"  Skipped queries: {skipped_count:,}")
    print(f"  File: {output_path}")
    return row_count


def train_model(train_path: Path):
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        "1",
        "-m",
        "FlagEmbedding.finetune.reranker.encoder_only.base",

        "--model_name_or_path",
        MODEL_NAME,

        "--train_data",
        str(train_path),

        "--output_dir",
        str(CHECKPOINTS_DIR),

        "--cache_path",
        str(CACHE_DIR / "data"),

        "--cache_dir",
        str(CACHE_DIR / "models"),

        "--train_group_size",
        str(TRAIN_GROUP_SIZE),

        "--query_max_len",
        str(QUERY_MAX_LENGTH),

        "--passage_max_len",
        str(PASSAGE_MAX_LENGTH),

        "--knowledge_distillation",
        "False",

        "--gradient_checkpointing",

        "--weight_decay",
        str(WEIGHT_DECAY),

        "--learning_rate",
        str(LEARNING_RATE),

        "--num_train_epochs",
        str(TRAIN_EPOCHS),

        "--per_device_train_batch_size",
        str(BATCH_SIZE),

        "--gradient_accumulation_steps",
        str(GRADIENT_ACCUMULATION_STEPS),

        "--fp16",

        "--dataloader_drop_last",
        "False",

        "--logging_steps",
        "10",

        "--save_strategy",
        "epoch",
    ]
    print("\nStarting training")
    print(f"Model: {MODEL_NAME}")
    print(f"Training data: {train_path}")
    print(f"Output directory: {CHECKPOINTS_DIR}")
    print("$ " + " ".join(command))
    subprocess.run(command, check=True)


def score_pairs(
    reranker,
    pairs: Sequence[tuple[str, str]],
    batch_size: int,
    max_length: int,
) -> list[float]:
    if not pairs:
        return []
    scores = reranker.compute_score(
        list(pairs),
        batch_size=batch_size,
        max_length=max_length,
    )
    if isinstance(scores, (int, float)):
        return [float(scores)]
    return [float(score) for score in scores]


def checkpoint_candidates(
    model_output: Path,
) -> list[tuple[str, Path]]:
    """Return saved Trainer checkpoints"""

    checkpoints = []
    for path in model_output.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort(key=lambda item: item[0])
    candidates = [(path.name, path) for _, path in checkpoints]
    if not candidates:
        raise FileNotFoundError(f"No saved checkpoints found in {model_output}")
    return candidates


def save_best_checkpoint(
    source: Path,
    destination: Path,
    source_label: str
):
    if destination.exists():
        shutil.rmtree(destination)

    if source.name.startswith("checkpoint-"):
        ignore = shutil.ignore_patterns(
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "events.out.tfevents*",
        )
    else:
        ignore = shutil.ignore_patterns(
            "checkpoint-*",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "events.out.tfevents*",
        )
    shutil.copytree(
        source,
        destination,
        ignore=ignore,
    )
    print(f"\nSelected checkpoint: {source_label}")
    print(f"Saved best model: {destination}")


def select_best_checkpoint(train_pairs, validation_pairs):
    checkpoints = checkpoint_candidates(CHECKPOINTS_DIR)
    best_validation_loss = float("inf")
    best_checkpoint = None
    history = []

    for epoch, (label, checkpoint) in enumerate(checkpoints, start=1):
        print(f"\nEvaluating checkpoint: {label}")
        print(f"Path: {checkpoint}")
        reranker = FlagReranker(str(checkpoint), use_fp16=True, devices=[DEVICE])
        try:
            train_loss = calculate_pairwise_loss(reranker=reranker, pairs=train_pairs)
            validation_loss = calculate_pairwise_loss(reranker=reranker, pairs=validation_pairs)
        finally:
            del reranker
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        result = {
            "epoch": epoch,
            "checkpoint": label,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        }
        history.append(result)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_checkpoint = (label, checkpoint)

    best_label, best_path = best_checkpoint
    save_best_checkpoint(
        source=best_path,
        destination=BEST_MODEL_DIR,
        source_label=best_label,
    )
    with open(OUTPUT_DIR / "training_history.json", "w") as file:
        json.dump(history, file, indent=2)


def build_pairs(
    topics: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]],
    corpus: Mapping[str, str],
    included_qids: set[str],
    seed: int = SEED,
    max_pairs_per_query: int = 100,
):
    rng = random.Random(seed)
    pairs = []
    for qid in sorted(included_qids):
        query = topics.get(qid, "").strip()
        if not query:
            continue
        judgments = qrels.get(qid, {})
        positives = [corpus[docid] for docid, grade in judgments.items() if grade >= 2 and docid in corpus and corpus[docid].strip()]
        negatives = [corpus[docid] for docid, grade in judgments.items() if grade == 1 and docid in corpus and corpus[docid].strip()]
        if not positives or not negatives:
            continue
        query_pairs = [(query, positive, negative) for positive in positives for negative in negatives]
        if len(query_pairs) > max_pairs_per_query:
            query_pairs = rng.sample(query_pairs, max_pairs_per_query,)
        pairs.extend(query_pairs)
    if not pairs:
        raise RuntimeError("No validation pairs were created")
    return pairs


def calculate_pairwise_loss(
    reranker,
    pairs,
    batch_size=BATCH_SIZE,
    max_length=MODEL_MAX_LENGTH,
):
    if not pairs:
        raise ValueError("Cannot calculate loss on an empty dataset.")

    total_loss = 0.0
    total_pairs = 0
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        scoring_pairs = []
        for query, positive, negative in chunk:
            scoring_pairs.append((query, positive))
            scoring_pairs.append((query, negative))
        scores = score_pairs(
            reranker,
            scoring_pairs,
            batch_size=batch_size * 2,
            max_length=max_length,
        )
        scores = np.asarray(scores, dtype=np.float64).reshape(-1, 2)
        positive_scores = scores[:, 0]
        negative_scores = scores[:, 1]
        losses = np.logaddexp(0.0, negative_scores - positive_scores,)
        total_loss += float(losses.sum())
        total_pairs += len(chunk)
    return total_loss / max(total_pairs, 1)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading topics, qrels, and paragraph corpus...")
    paragraph_corpus = load_corpus(PARAGRAPH_CORPUS)

    with open(TRAIN_TOPICS, "r", encoding="utf-8") as file:
        data = json.load(file)
    topics = {}
    for sample in data:
        qid = sample["query_id"]
        text = sample["query"]
        topics[str(qid)] = str(text).strip()
    training_topics = {qid: text for qid, text in topics.items() if text}

    qrels = {}
    if not PARAGRAPH_QRELS.exists():
        raise FileNotFoundError(f"Qrels file not found: {PARAGRAPH_QRELS}")

    with PARAGRAPH_QRELS.open("r", encoding="utf-8") as file:
        for line in file:
            fields = line.strip().split()
            if str(fields[0]) not in qrels:
                qrels[str(fields[0])] = {}
            qrels[str(fields[0])][str(fields[2])] = float(fields[3])

    print(f"Training topics: {len(training_topics):,}")
    print(f"Paragraphs in corpus: {len(paragraph_corpus):,}")
    print(f"Paragraph qrels queries: {len(qrels):,}")

    train_qids, validation_qids = split_training_queries(
        topics=training_topics,
        qrels=qrels,
        corpus=paragraph_corpus,
        validation_size=VALIDATION_SIZE,
        seed=SEED,
    )
    fraction_tag = f"{VALIDATION_SIZE:g}".replace(".", "p")
    train_path = TRAINING_DIR / f"grade2_vs_grade1_seed{SEED}_val{fraction_tag}.jsonl"
    validation_qids_path = TRAINING_DIR / f"validation_qids_seed{SEED}_val{fraction_tag}.json"

    if not train_path.exists():
        build_training_jsonl(
            topics=training_topics,
            qrels=qrels,
            corpus=paragraph_corpus,
            included_qids=train_qids,
            output_path=train_path,
        )
    else:
        print(f"\nUsing existing training JSONL: {train_path}")
    with open(validation_qids_path, "w") as file:
        json.dump(sorted(validation_qids), file, indent=2)

    train_model(train_path=train_path)
    train_pairs = build_pairs(
        topics=training_topics,
        qrels=qrels,
        corpus=paragraph_corpus,
        included_qids=train_qids,
        seed=SEED,
    )
    validation_pairs = build_pairs(
        topics=training_topics,
        qrels=qrels,
        corpus=paragraph_corpus,
        included_qids=validation_qids,
        seed=SEED + 1,
    )
    print("\nSelecting best checkpoint...")
    select_best_checkpoint(train_pairs, validation_pairs)


if __name__ == "__main__":
    main()
