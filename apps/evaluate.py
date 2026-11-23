"""
Run each prompt-map approach on Study and Test and score Study.

Usage:
    python -m apps.evaluate
"""

import json
import time
from itertools import combinations
from tqdm import tqdm

from ranx import Qrels, Run, compare, evaluate

from apps.retrieval import retrieve
from apps.utils import DATA_DIR, EVAL_DIR, PROMPT_MAP, QRELS_DIR, ROOT_DIR

RUN_DIR = DATA_DIR / "runs"
QUERIES_DIR = DATA_DIR / "queries"
LEVELS = ("paper", "paragraph")
K = 100
METRICS = ["ndcg", "ndcg@5", "mrr", "precision@5", "precision@10"]
STRICT_METRICS = ["mrr", "precision@5"]

def load_queries(name):
    rows = json.loads((DATA_DIR / f"{name.capitalize()}.json").read_text())
    return [(str(row["query_id"]), row["query"]) for row in rows]

def evaluate_level(name, level, queries, generated):
    runs, paths, latency = {}, {}, {}
    for method, (run, per_query) in generated.items():
        runs[method] = run
        paths[method] = str((RUN_DIR / f"TheSearchParty_{level}_{method}_{name}.tsv").relative_to(ROOT_DIR))
        latency[method] = per_query

    file = RUN_DIR / f"TheSearchParty_{level}_bm25_{name}.tsv"
    runs["bm25"] = Run.from_file(str(file), kind="trec", name="bm25")
    paths["bm25"] = str(file.relative_to(ROOT_DIR))

    prefix = f"{name}_{level}"
    means = {}
    for method, per_query in latency.items():
        ordered = [per_query[qid] for qid in sorted(per_query, key=int)]
        kept = ordered[1:] or ordered
        means[method] = sum(kept) / len(kept)

    performance = {"latency_ms": latency, "latency_mean_ms": means}

    metric_passes, query_passes, significance = {}, {}, {}

    for key, minimum, metrics in (
        ("relevant_or_partial", 1, METRICS),
        ("relevant_only", 2, STRICT_METRICS),
    ):
        file = QRELS_DIR / f"{name.capitalize()}_{level}_qrel.tsv"
        judged = Qrels.from_file(str(file), kind="trec").to_dict()
        qrels = Qrels.from_dict({qid: judged[qid] for qid, _ in queries if qid in judged})
        qrels.set_relevance_level(rel_lvl=minimum)

        scores = {}
        detailed = {qid: {} for qid, _ in queries}

        for method, run in runs.items():
            values = evaluate(
                qrels, run, metrics,
                make_comparable=True, save_results_in_run=True,
            )
            scores[method] = {metric: float(values[metric]) for metric in metrics}

            for metric in metrics:
                for qid, score in run.scores[metric].items():
                    if qid in detailed:
                        detailed[qid].setdefault(method, {})[metric] = float(score)

            print(f"  {name}/{level}/{key} {method}: {scores[method]}", flush=True)

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
                comparisons[f"{left} vs {right}"] = {
                    metric: {
                        "p": float(f"{pvals[metric]:.3g}"),
                        "better": (
                            left if scores[left][metric] > scores[right][metric] else right
                        ) if pvals[metric] < 0.05 else None,
                        "win_tie_loss": dict(counts[metric]),
                    }
                    for metric in metrics
                }
            significance[key] = {"relevance_level": minimum, "comparisons": comparisons}

    context = {"set": name, "level": level, "stem": True, "k": K}
    summary = {
        **context,
        "performance": performance,
        "trec_runs": paths,
        "relevance_passes": metric_passes,
        "queries_file": str((EVAL_DIR / f"{prefix}_queries.json").relative_to(ROOT_DIR)),
        "significance_file": str((EVAL_DIR / f"{prefix}_significance.json").relative_to(ROOT_DIR)),
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / f"{prefix}.json").write_text(json.dumps(summary, indent=2) + "\n")
    (EVAL_DIR / f"{prefix}_queries.json").write_text(
        json.dumps({**context, "relevance_passes": query_passes}, indent=2) + "\n"
    )
    (EVAL_DIR / f"{prefix}_significance.json").write_text(json.dumps({
        **context, "stat_test": "student", "max_p": 0.05,
        "relevance_passes": significance,
    }, indent=2) + "\n")
    print(f"Saved metrics: {EVAL_DIR / f'{prefix}.json'}", flush=True)


def main():
    sets = {"study": load_queries("study"), "test": load_queries("test")}
    for level in LEVELS:
        file = QRELS_DIR / f"Study_{level}_qrel.tsv"
        if not file.exists():
            raise FileNotFoundError(file)

    generated = {name: {level: {} for level in LEVELS} for name in sets}
    approaches = [("qwen", None), *[(name, name) for name in PROMPT_MAP]]

    for label, approach in approaches:
        for name, queries in sets.items():
            for level in LEVELS:
                gen_queries, ranked, latency = {}, {}, {}
                for qid, text in tqdm(queries, desc=f"Generating queries for {name}/{level}/{label}"):
                    start = time.perf_counter()
                    query, results = retrieve(text, approach=approach, level=level, k=K)
                    ranked[qid] = dict(results)
                    gen_queries[qid] = query
                    latency[qid] = (time.perf_counter() - start) * 1000

                run = Run.from_dict(ranked, name=label)
                generated[name][level][label] = (run, latency)

                file = RUN_DIR / f"TheSearchParty_{level}_{label}_{name}.tsv"
                file.parent.mkdir(parents=True, exist_ok=True)
                run.save(str(file), kind="trec")
                print(f"Saved run: {file.relative_to(ROOT_DIR)}", flush=True)

                queries_file = QUERIES_DIR / f"queries_{level}_{label}_{name}.json"
                queries_file.parent.mkdir(parents=True, exist_ok=True)
                with open(queries_file, "w") as file:
                    json.dump(gen_queries, file, indent=2)
                print(f"Saved queries: {queries_file.relative_to(ROOT_DIR)}", flush=True)

    for level in LEVELS:
        evaluate_level("study", level, sets["study"], generated["study"][level])

if __name__ == "__main__":
    main()
