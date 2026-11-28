"""
Run each generated approach on Study and Test and score Study.

Usage:
    python -m apps.evaluate
"""

import json
import time
from itertools import combinations
from tqdm import tqdm

from ranx import Qrels, Run, compare, evaluate

from apps.retrieval import retrieve
from apps.utils import DATA_DIR, GENERATED_DIR, METRICS_DIR, PROMPT_MAP, QRELS_DIR, ROOT_DIR

RUN_DIR = DATA_DIR / "runs"
LEVELS = ("paper", "paragraph")
K = 100
# Full metric set at relevance >= 1. The strict pass keeps MRR and P@5 at relevance >= 2.
METRICS = ["ndcg", "ndcg@5", "mrr", "precision@5", "precision@10"]
STRICT_METRICS = ["mrr", "precision@5"]

def evaluate_level(level, queries, generated):
    # Study runs, their TREC paths, and the times collected while retrieving.
    runs, paths, retrieval = {}, {}, {}
    for method, record in generated.items():
        runs[method] = record["run"]
        paths[method] = record["path"]
        retrieval[method] = record["retrieval_ms"]

    # Retrieval time for every method. The mean leaves out the lowest query id, which loads the index.
    prefix = f"study_{level}"
    retrieval_mean_ms = {}
    for method, times in retrieval.items():
        ordered = [times[qid] for qid in sorted(times, key=int)]
        kept = ordered[1:] or ordered
        retrieval_mean_ms[method] = sum(kept) / len(kept)
    performance = {"retrieval_ms": retrieval, "retrieval_mean_ms": retrieval_mean_ms}

    # Judged Study queries, loaded once. Each pass copies them and applies its cutoff.
    file = QRELS_DIR / f"Study_{level}_qrel.tsv"
    judged = Qrels.from_file(str(file), kind="trec").to_dict()
    judged = {qid: judged[qid] for qid, _ in queries if qid in judged}

    metric_passes, query_passes, significance = {}, {}, {}

    # One pass per relevance cutoff: partial-or-relevant, then relevant only.
    for key, minimum, metrics in (
        ("relevant_or_partial", 1, METRICS),
        ("relevant_only", 2, STRICT_METRICS),
    ):
        qrels = Qrels.from_dict({qid: dict(docs) for qid, docs in judged.items()})
        qrels.set_relevance_level(rel_lvl=minimum)

        scores = {}
        detailed = {qid: {} for qid, _ in queries}

        for method, run in runs.items():
            # Aggregate metrics, and per-query scores saved back onto the run.
            values = evaluate(
                qrels, run, metrics,
                make_comparable=True, save_results_in_run=True,
            )
            scores[method] = {metric: float(values[metric]) for metric in metrics}

            for metric in metrics:
                for qid, score in run.scores[metric].items():
                    if qid in detailed:
                        detailed[qid].setdefault(method, {})[metric] = float(score)

            print(f"  study/{level}/{key} {method}: {scores[method]}", flush=True)

        metric_passes[key] = {"relevance_level": minimum, "metrics": scores}
        query_passes[key] = {
            "relevance_level": minimum,
            "queries": {qid: scores for qid, scores in detailed.items() if scores},
        }

        # Pairwise Student t-test across the saved systems.
        if len(runs) > 1:
            report = compare(
                qrels, list(runs.values()), metrics,
                stat_test="student", max_p=0.05, make_comparable=True,
            ).to_dict()

            comparisons = {}
            for left, right in combinations(runs, 2):
                pvals = report[left]["comparisons"][right]
                counts = report[left]["win_tie_loss"][right]
                pair = {}
                for metric in metrics:
                    higher = left if scores[left][metric] > scores[right][metric] else right
                    pair[metric] = {
                        "p": float(f"{pvals[metric]:.3g}"),
                        "better": higher if pvals[metric] < 0.05 else None,
                        "win_tie_loss": dict(counts[metric]),
                    }
                comparisons[f"{left} vs {right}"] = pair
            significance[key] = {"relevance_level": minimum, "comparisons": comparisons}

    # Summary, per-query scores, and pairwise tests, each in its own file.
    context = {"set": "study", "level": level, "stem": True, "k": K}
    summary = {
        **context,
        "performance": performance,
        "trec_runs": paths,
        "relevance_passes": metric_passes,
        "queries_file": str((METRICS_DIR / f"{prefix}_queries.json").relative_to(ROOT_DIR)),
        "significance_file": str((METRICS_DIR / f"{prefix}_significance.json").relative_to(ROOT_DIR)),
    }
    payloads = {
        f"{prefix}.json": summary,
        f"{prefix}_queries.json": {**context, "relevance_passes": query_passes},
        f"{prefix}_significance.json": {
            **context, "stat_test": "student", "max_p": 0.05,
            "relevance_passes": significance,
        },
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, payload in payloads.items():
        (METRICS_DIR / filename).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved metrics: {METRICS_DIR / f'{prefix}.json'}", flush=True)


def main():
    # Study is scored. Test is retrieved only so its TREC run can be saved.
    for level in LEVELS:
        file = QRELS_DIR / f"Study_{level}_qrel.tsv"
        if not file.exists():
            raise FileNotFoundError(file)

    study = {level: {} for level in LEVELS}
    study_queries = None
    for approach in PROMPT_MAP:
        for name in ("study", "test"):
            file = GENERATED_DIR / f"{name}_{approach}.json"
            if not file.is_file():
                raise FileNotFoundError(file)
            replies = json.loads(file.read_text())["queries"]
            queries = [(str(qid), text) for qid, text in replies.items()]
            if name == "study":
                study_queries = queries
            for level in LEVELS:
                # Time retrieval on Study only. Both sets write a TREC run.
                ranked, latency = {}, {}
                for qid, reply in tqdm(queries, desc=f"Retrieving {name}/{level}/{approach}"):
                    if name == "study":
                        start = time.perf_counter()
                    ranked[qid] = dict(retrieve(
                        reply, approach=approach, original=reply, level=level, k=K,
                    ))
                    if name == "study":
                        latency[qid] = (time.perf_counter() - start) * 1000

                run = Run.from_dict(ranked, name=approach)
                file = RUN_DIR / f"TheSearchParty_{level}_{approach}_{name}.tsv"
                file.parent.mkdir(parents=True, exist_ok=True)
                run.save(str(file), kind="trec")
                print(f"Saved run: {file.relative_to(ROOT_DIR)}", flush=True)

                # Study keeps the run and its retrieval times.
                if name != "study":
                    continue
                study[level][approach] = {
                    "run": run,
                    "path": str(file.relative_to(ROOT_DIR)),
                    "retrieval_ms": latency,
                }

    # Metrics, significance, and query times are written for Study only.
    for level in LEVELS:
        evaluate_level(level, study_queries, study[level])

if __name__ == "__main__":
    main()
