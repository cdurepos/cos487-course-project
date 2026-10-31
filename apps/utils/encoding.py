
"""Shared, differentiable chunk encoding for SPLADE and SBERT."""
import torch
import torch.nn.functional as F

def encode(model, texts, pooling="max", batch_size=8, max_tokens=None, normalize=False, return_lengths=False):
    """Encode text(s) using chunk-wise max or token-weighted mean pooling."""

    if pooling not in ("max", "mean"):
        raise ValueError("pooling must be 'max' or 'mean'")

    single = isinstance(texts, str)
    texts = [texts] if single else list(texts)
    if not texts:
        raise ValueError("No texts to encode")

    tok = model.tokenizer
    limit = max_tokens or model.max_seq_length
    if not limit:
        raise ValueError("Specify max_tokens for models without a sequence limit")

    # Infer native special-token prefix/suffix (BERT, RoBERTa, MPNet, etc.).
    template = tok("example", add_special_tokens=True, return_special_tokens_mask=True)
    mask = template["special_tokens_mask"]
    first, last = mask.index(0), len(mask) - mask[::-1].index(0)
    prefix = template["input_ids"][:first]
    suffix = template["input_ids"][last:]
    budget = limit - len(prefix) - len(suffix)
    if budget < 1:
        raise ValueError("max_tokens is too small for the tokenizer")

    token_ids = tok(texts, add_special_tokens=False, verbose=False)["input_ids"]
    lengths = [len(ids) for ids in token_ids]
    chunks, owners, weights = [], [], []
    for j, ids in enumerate(token_ids):
        for start in range(0, len(ids), budget):
            part = ids[start:start + budget]
            chunks.append(prefix + part + suffix)
            owners.append(j)
            weights.append(len(part))

    outputs = [None] * len(texts)
    for start in range(0, len(chunks), batch_size):
        batch = tok.pad({"input_ids": chunks[start:start + batch_size]},
                        return_tensors="pt").to(model.device)
        vectors = model(batch)["sentence_embedding"]
        for j, vector, weight in zip(owners[start:start + batch_size], vectors,
                                      weights[start:start + batch_size]):
            value = vector if pooling == "max" else vector * weight
            outputs[j] = value if outputs[j] is None else (
                torch.maximum(outputs[j], value) if pooling == "max"
                else outputs[j] + value
            )

    zero = torch.zeros(model.get_embedding_dimension(), device=model.device)
    result = torch.stack([
        (v / lengths[j] if pooling == "mean" else v) if v is not None else zero
        for j, v in enumerate(outputs)
    ])
    if normalize:
        result = F.normalize(result, p=2, dim=-1)
    if single:
        result, lengths = result[0], lengths[0]
    return (result, lengths) if return_lengths else result
