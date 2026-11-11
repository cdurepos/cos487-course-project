"""
Fine-tune a qlora adapter for the Qwen3-Embedding-0.6B model

Usage:
    python -m apps.finetuning.dense
"""


import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
INSTRUCTION = "Given a scientific research question, retrieve paragraphs from scientific papers that are relevant to answering the question."

TOPICS = "data/Train.json"
QRELS = "data/qrels/Train_paragraph_qrel.tsv"
CORPUS_DIR = "data/JSON Files"
OUTPUT_DIR = "models/qwen"
SEED = 42

EPOCHS = 10
BATCH_SIZE = 4
GRAD_ACCUM = 4
MAX_LENGTH = 512
LEARNING_RATE = 1e-5
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


prepare_query = lambda query: f"Instruct: {INSTRUCTION}\nQuery: {query}"


def iter_corpus(corpus_dir):
    """Yield (paragraph_id, text) from corpus JSON files"""
    for path in os.listdir(corpus_dir):
        if not path.endswith(".json"):
            continue
        try:
            with open(os.path.join(corpus_dir, path), "r") as file:
                data = json.load(file)
        except Exception as e:
            print(f"[warn] Skipping unreadable JSON {path}: {e}")
            continue

        paragraphs = data.get("paragraphs", {})
        items = paragraphs.items()
        for pid, text in items:
            yield str(pid), text


def load_positive_texts(corpus_dir, needed_ids):
    """
    Load judged documents from the corpus
    Missing document IDs are reported and skipped
    """
    found = {}
    for pid, text in iter_corpus(corpus_dir):
        if pid in needed_ids:
            found[pid] = text
            if len(found) == len(needed_ids):
                break
    missing = needed_ids - set(found)
    if missing:
        print(f"[warn] {len(missing)} judged-positive documents were not found in the corpus. They will be skipped. First missing IDs: {sorted(missing)[:20]}")
    print(f"[data] Found {len(found)} / {len(needed_ids)} judged-positive documents.")
    return found, missing


def split_query_ids(query_ids, seed, train_frac=0.9):
    ids = list(query_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_train = max(1, int(len(ids) * train_frac))
    return ids[:n_train], ids[n_train:]


def load_quantized_model(model_name, use_4bit=True):
    compute_dtype = torch.float16
    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=quant_config,
        torch_dtype=compute_dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    return model, tokenizer


def add_lora(model, rank=16, alpha=32, dropout=0.05):
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION",
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def last_token_embeddings(model, tokenizer, texts, max_length, device):
    """Pool the final non-padding token, then normalize"""
    if not texts:
        hidden_size = model.config.hidden_size
        return torch.empty((0, hidden_size), device=device, dtype=torch.float32)
    batch = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch, return_dict=True)
    hidden = outputs.last_hidden_state
    last_idx = batch["attention_mask"].sum(dim=1) - 1
    row_idx = torch.arange(hidden.shape[0], device=device)
    pooled = hidden[row_idx, last_idx]
    return F.normalize(pooled.float(), p=2, dim=-1)


@torch.no_grad()
def encode_texts(
    model,
    tokenizer,
    texts,
    max_length,
    batch_size,
    device,
):
    model.eval()
    out = []

    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        with torch.autocast("cuda", dtype=torch.float16):
            emb = last_token_embeddings(
                model,
                tokenizer,
                chunk,
                max_length,
                device,
            )
        out.append(emb.cpu().numpy().astype(np.float32))
    if not out:
        return np.empty((0, model.config.hidden_size), dtype=np.float32)
    return np.concatenate(out, axis=0)


def make_diverse_batches(query_ids, query_vectors, batch_size, seed):
    """Greedy farthest-point batching using query embeddings from pretrained model"""
    rng = np.random.default_rng(seed)
    ids = list(query_ids)

    vecs = np.asarray([query_vectors[q] for q in ids], dtype=np.float32)
    vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    sim = vecs @ vecs.T
    remaining = set(range(len(ids)))
    batches = []

    while remaining:
        first = int(rng.choice(list(remaining)))
        chosen = [first]
        remaining.remove(first)
        max_sim = sim[:, first].copy()
        while remaining and len(chosen) < batch_size:
            candidates = np.fromiter(remaining, dtype=np.int64)
            nxt = int(candidates[np.argmin(max_sim[candidates])])
            chosen.append(nxt)
            remaining.remove(nxt)
            max_sim = np.maximum(max_sim, sim[:, nxt])
        batches.append([ids[i] for i in chosen])
    return batches


def multi_positive_infonce(
    query_emb,
    doc_emb,
    candidate_pids,
    qids,
    qrels,
    temperature=0.05,
):
    """
    Every candidate known positive for a query is marked positive
    Other candidate documents are treated as unjudged negatives
    """
    if query_emb.shape[0] == 0 or doc_emb.shape[0] == 0:
        return query_emb.sum() * 0.0
    
    scores = (query_emb @ doc_emb.T) / temperature
    positive = torch.zeros_like(scores, dtype=torch.bool)
    for i, qid in enumerate(qids):
        known = qrels[qid]
        for j, pid in enumerate(candidate_pids):
            if pid in known:
                positive[i, j] = True

    has_positive = positive.any(dim=1)
    if not has_positive.any():
        return scores.sum() * 0.0

    log_den = torch.logsumexp(scores, dim=1)
    log_num = torch.logsumexp(scores.masked_fill(~positive, -torch.inf), dim=1,)
    loss_per_query = log_den - log_num
    return loss_per_query[has_positive].mean()


def train_one_epoch(
    model,
    tokenizer,
    optimizer,
    scaler,
    train_qids,
    qrels,
    pid_to_text,
    query_texts,
    query_vectors,
    epoch,
):
    model.train()
    batches = make_diverse_batches(
        train_qids,
        query_vectors,
        BATCH_SIZE,
        SEED + epoch,
    )
    rng = random.Random(SEED + 1000 + epoch)
    optimizer.zero_grad(set_to_none=True)

    epoch_loss_sum = 0.0
    epoch_query_count = 0
    step_loss_sum = 0.0
    step_query_count = 0
    updates = 0
    device = next(model.parameters()).device

    for step, batch_qids in enumerate(batches):
        selected = [rng.choice(list(qrels[qid].keys())) for qid in batch_qids]
        candidate_pids = list(dict.fromkeys(selected))
        doc_inputs = [pid_to_text[pid] for pid in candidate_pids]
        query_inputs = [prepare_query(query_texts[qid]) for qid in batch_qids]

        with torch.autocast("cuda", dtype=torch.float16):
            q_emb = last_token_embeddings(
                model,
                tokenizer,
                query_inputs,
                MAX_LENGTH,
                device,
            )
            d_emb = last_token_embeddings(
                model,
                tokenizer,
                doc_inputs,
                MAX_LENGTH,
                device,
            )
            loss = multi_positive_infonce(
                q_emb,
                d_emb,
                candidate_pids,
                batch_qids,
                qrels,
            )
            raw_loss = loss.detach().item()
            scaled_loss = loss / GRAD_ACCUM

        scaler.scale(scaled_loss).backward()
        step_loss_sum += raw_loss * len(batch_qids)
        step_query_count += len(batch_qids)
        epoch_loss_sum += raw_loss * len(batch_qids)
        epoch_query_count += len(batch_qids)

        is_update_step = ((step + 1) % GRAD_ACCUM == 0 or step + 1 == len(batches))
        if is_update_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            optimizer_step_loss = (
                step_loss_sum / max(1, step_query_count)
            )
            print(f"[train-step] epoch={epoch} step={updates}/{math.ceil(len(batches) / GRAD_ACCUM)} loss={optimizer_step_loss:.6f}")
            step_loss_sum = 0.0
            step_query_count = 0

    epoch_loss = epoch_loss_sum / max(1, epoch_query_count)
    print(f"[train-epoch] epoch={epoch} average_loss={epoch_loss:.6f}")
    return epoch_loss, updates


@torch.no_grad()
def evaluate_validation_loss(
    model,
    tokenizer,
    val_qids,
    val_qrels,
    pid_to_text,
    query_texts,
    epoch,
):
    model.eval()

    total_loss = 0.0
    total_queries = 0
    batches = [val_qids[i:i + BATCH_SIZE] for i in range(0, len(val_qids), BATCH_SIZE)]

    device = next(model.parameters()).device
    for step, qids in enumerate(batches, 1):
        selected = [sorted(val_qrels[qid].keys())[0] for qid in qids]
        candidate_pids = list(dict.fromkeys(selected))
        query_inputs = [prepare_query(query_texts[qid]) for qid in qids]
        doc_inputs = [pid_to_text[pid] for pid in candidate_pids]

        with torch.autocast("cuda", dtype=torch.float16):
            q_emb = last_token_embeddings(
                model,
                tokenizer,
                query_inputs,
                MAX_LENGTH,
                device
            )
            d_emb = last_token_embeddings(
                model,
                tokenizer,
                doc_inputs,
                MAX_LENGTH,
                device
            )
            loss = multi_positive_infonce(
                q_emb,
                d_emb,
                candidate_pids,
                qids,
                val_qrels
            )
        batch_loss = loss.item()
        total_loss += batch_loss * len(qids)
        total_queries += len(qids)
        print(f"[validation-step] epoch={epoch} step={step}/{len(batches)} loss={batch_loss:.6f}")

    average_loss = total_loss / max(1, total_queries)
    print(f"[validation-epoch] epoch={epoch} average_loss={average_loss:.6f}")
    return average_loss


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this QLoRA script.")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(TOPICS, "r") as file:
        topic_data = json.load(file)
    topics = {str(query["query_id"]): query["query"] for query in topic_data}

    with open(QRELS, "r") as file:
        qrels_all = {}
        for line in file:
            query_id, _, para_id, level = line.strip().split("\t")
            if query_id not in topics:
                continue
            if query_id not in qrels_all:
                qrels_all[query_id] = {}
            qrels_all[query_id][para_id] = int(level)

    needed_ids = set()
    for qid in qrels_all:
        needed_ids.update(qrels_all[qid].keys())
    print(f"[data] Found {len(qrels_all)} queries and {len(needed_ids)} documents.")

    pid_to_text, missing_ids = load_positive_texts(CORPUS_DIR, needed_ids)
    qrels_available = {}
    for query_id in qrels_all:
        available_rels = {para_id: grade for para_id, grade in qrels_all[query_id].items() if para_id in pid_to_text}
        if available_rels:
            qrels_available[query_id] = available_rels

    train_qids, val_qids = split_query_ids(qrels_available.keys(), SEED)
    train_qrels = {qid: qrels_available[qid] for qid in train_qids}
    val_qrels = {qid: qrels_available[qid] for qid in val_qids}
    query_texts = {qid: topics[qid] for qid in qrels_available.keys()}

    print(f"[data] Queries train/validation: {len(train_qids)}/{len(val_qids)}")
    print(f"[data] Training judged-positive pairs: {sum(len(x) for x in train_qrels.values())}")
    print(f"[data] Validation judged-positive pairs: {sum(len(x) for x in val_qrels.values())}")
    print(f"[data] Missing judged document IDs: {len(missing_ids)}")

    model, tokenizer = load_quantized_model(MODEL_NAME, use_4bit=True)
    model = add_lora(model, LORA_RANK, LORA_ALPHA, LORA_DROPOUT)
    device = next(model.parameters()).device

    print("[batching] Computing query embeddings with pretrained model...")
    all_train_query_inputs = [prepare_query(query_texts[qid]) for qid in train_qids]
    base_vecs = encode_texts(
        model,
        tokenizer,
        all_train_query_inputs,
        MAX_LENGTH,
        BATCH_SIZE,
        device,
    )
    query_vectors = {qid: base_vecs[i] for i, qid in enumerate(train_qids)}

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    best_validation_loss = float("inf")
    best_adapter_dir = (Path(OUTPUT_DIR) / "best_adapter")
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, updates = train_one_epoch(
            model,
            tokenizer,
            optimizer,
            scaler,
            train_qids,
            train_qrels,
            pid_to_text,
            query_texts,
            query_vectors,
            epoch,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "optimizer_updates": updates,
        }
        print(f"[epoch {epoch}] train_loss={train_loss:.6f} updates={updates}")
        val_loss = evaluate_validation_loss(
            model,
            tokenizer,
            val_qids,
            val_qrels,
            pid_to_text,
            query_texts,
            epoch,
        )
        row["validation_loss"] = val_loss
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            model.save_pretrained(best_adapter_dir)
            tokenizer.save_pretrained(best_adapter_dir)
            print(f"[checkpoint] New best validation checkpoint: epoch={epoch}, validation_loss={best_validation_loss:.5f}")
        history.append(row)
        with open(Path(OUTPUT_DIR) / "training_history.json", "w") as file:
            json.dump(history, file, indent=2)
            
        latest_dir = Path(OUTPUT_DIR) / "latest"
        model.save_pretrained(latest_dir)
        tokenizer.save_pretrained(latest_dir)

    print(f"[done] Outputs saved under: {OUTPUT_DIR}")
    print(f"[done] Best retrieval adapter: {best_adapter_dir}")


if __name__ == "__main__":
    main()
