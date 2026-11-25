"""Generate each prompt once for Study and Test and save the replies.

Usage:
    python -m apps.generate
"""

import json
import time

import torch
from tqdm import tqdm

from apps.gemma import generate, load_model
from apps.utils import DATA_DIR, GENERATED_DIR, METRICS_DIR, PROMPT_MAP, ROOT_DIR

SEED = 42
SETS = ("study", "test")
I = 3

if __name__ == "__main__":
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model, assistant, processor = load_model()
    tokenizer = getattr(processor, "tokenizer", processor)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    for approach, prompt in PROMPT_MAP.items():
        for name in SETS:
            rows = json.loads((DATA_DIR / f"{name.capitalize()}.json").read_text())
            questions = [(str(row["query_id"]), row["query"]) for row in rows]
            saved, timings = {}, {}
            for qid, question in tqdm(questions, desc=f"{name}/{approach}"):
                # Study timings are metrics. Test only needs the reply text.
                if name == "study":
                    start = time.perf_counter()
                text = generate(
                    messages=[{"role": "user", "content": prompt.format(query=question, i=I)}],
                    model=model,
                    processor=processor,
                    assistant=assistant,
                    max_new_tokens=512,
                )
                saved[qid] = text
                if name == "study":
                    ms = (time.perf_counter() - start) * 1000
                    tokens = len(tokenizer.encode(text, add_special_tokens=False))
                    timings[qid] = {
                        "ms": ms,
                        "tokens": tokens,
                        "time_per_token": ms / tokens if tokens else None,
                    }

            file = GENERATED_DIR / f"{name}_{approach}.json"
            file.write_text(json.dumps({
                "seed": SEED,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 64,
                "queries": saved,
            }, indent=2) + "\n")
            print(f"Saved {file.relative_to(ROOT_DIR)}", flush=True)

            if name != "study":
                continue
            # Both means leave out the lowest query id, which loads the model.
            kept = sorted(timings, key=int)[1:] or list(timings)
            kept_ms = [timings[qid]["ms"] for qid in kept]
            kept_rates = [timings[qid]["time_per_token"] for qid in kept if timings[qid]["time_per_token"] is not None]
            metrics = METRICS_DIR / f"study_{approach}_generation.json"
            metrics.write_text(json.dumps({
                "set": "study",
                "approach": approach,
                **timings,
                "mean_ms": sum(kept_ms) / len(kept_ms),
                "mean_time_per_token": sum(kept_rates) / len(kept_rates) if kept_rates else None,
            }, indent=2) + "\n")
            print(f"Saved {metrics.relative_to(ROOT_DIR)}", flush=True)
