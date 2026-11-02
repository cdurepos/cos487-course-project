
"""Evaluate configured dense/sparse encoders against their saved CSR indexes.

Run: python -m apps.evaluation.evaluate_part2 [config.yaml]
"""
import json
import statistics
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import yaml
from ranx import Qrels, Run, compare, evaluate
from sentence_transformers import SentenceTransformer, SparseEncoder

from apps.utils.encoding import encode
from apps.utils.sharding import search

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RUNS = DATA / "runs"
OUT = DATA / "evaluation"
DEFAULT = Path(__file__).with_name("config.yaml")
METRICS = ["ndcg", "ndcg@5", "mrr", "precision@5", "precision@10"]
STRICT = ["mrr", "precision@5"]

def path(value):
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT / p


def write(filename, content):
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")


def queries_for(name, limit=None):
    rows = json.loads((DATA / f"{name.capitalize()}.json").read_text())
    rows = rows[:limit] if limit else rows
    return [(str(row["query_id"]), row["query"]) for row in rows]


def qrels_for(name, level, qids, minimum):
    filename = DATA / "qrels" / f"{name.capitalize()}_{level}_qrel.tsv"
    judged = Qrels.from_file(str(filename), kind="trec").to_dict()
    qrels = Qrels.from_dict({qid: docs for qid, docs in judged.items() if qid in qids})
    qrels.set_relevance_level(rel_lvl=minimum)
    return qrels


def run_path(team, level, method, set_name, stem=True):
    suffix = "" if stem else "_no-stem"
    return RUNS / f"{team}_{level}_{method}_{set_name}{suffix}.tsv"


def retrieve(model, spec, index, queries, levels, k):
    """Encode queries once. Report compute latency without shard-loading I/O."""
    is_sparse = spec["type"] == "sparse"
    vectors, encoding_ms = [], []

    with torch.inference_mode():
        for _, text in queries:
            begin = time.perf_counter()
            v = encode(
                model, text,
                pooling=spec.get("pooling", "max" if is_sparse else "mean"),
                normalize=spec.get("normalize", not is_sparse),
                max_tokens=spec.get("max_tokens", model.max_seq_length),
                batch_size=spec.get("batch_size", 8)
            )
            vectors.append(v.float().cpu().numpy())
            encoding_ms.append((time.perf_counter() - begin) * 1000)

    answers = {}
    for level in levels:
        begin = time.perf_counter()
        #
        results, scoring_ms = search(
            index,
            level + "s",
            np.stack(vectors),
            k,
            timed=True,
            normalized=spec.get("normalize_docs", False)
        )
        #
        elapsed = (time.perf_counter() - begin) * 1000

        samples = [e + s for e, s in zip(encoding_ms, scoring_ms)][1:]
        latency = {
            "queries": len(samples),
            "warmup_excluded": 1,
            "mean": round(statistics.mean(samples), 3) if samples else None,
            "median": round(statistics.median(samples), 3) if samples else None,
            "measurement": "query encoding + scoring (excludes shard disk I/O)",
            "shard_search_wall_ms": round(elapsed, 3),
        }

        files = [
            *index.glob(f"{level}s_*.npz"),
            *index.glob(f"{level}s_*_ids.json")
        ]
        info = {
            "path": str(index.relative_to(ROOT)) if index.is_relative_to(ROOT) else str(index),
            "bytes": sum(p.stat().st_size for p in files),
            "files": len(files)
        }

        ranked = {
            qid: dict(hits)
            for (qid, _), hits in zip(queries, results)
        }
        answers[level] = ranked, latency, info

    return answers


def evaluate_level(cfg, job, level, queries, generated):
    name, team = job["name"], cfg["team_name"]
    stem, k = cfg.get("stem", True), int(cfg.get("k", 100))
    methods = (
        cfg.get("comparison_methods", cfg.get("baseline_methods", []))
        if job["metrics"] else []
    )

    runs, paths = [], {}
    for method in [*methods, *cfg["models"]]:
        filename = run_path(
            team, level, method, name,
            stem if method in ("bm25", "tfidf") else True
        )
        if not filename.is_file():
            raise FileNotFoundError(f"Missing comparison run: {filename}")

        runs.append(Run.from_file(str(filename), kind="trec", name=method))
        paths[method] = str(filename.relative_to(ROOT))

    if not job["metrics"]:
        return

    query_ids = [qid for qid, _ in queries]
    context = {"set": name, "level": level, "stem": stem, "k": k}
    neural = {
        label: generated[(name, level, label)]
        for label in cfg["models"]
    }

    performance = {
        "indexes": {label: values[2] for label, values in neural.items()},
        "latency_ms": {label: values[1] for label, values in neural.items()},
    }
    if len(neural) == 1:
        performance["index"] = next(iter(performance["indexes"].values()))

    # Optionally retain historical Part 1 performance measurements.
    suffix = "" if stem else "_no-stem"
    prior = DATA / "evaluation" / f"{name}_{level}{suffix}.json"

    if prior.is_file():
        previous = json.loads(prior.read_text()).get("performance", {})
        performance["latency_ms"].update({
            m: previous["latency_ms"][m]
            for m in methods
            if m in previous.get("latency_ms", {})
        })
        if "index" in previous:
            performance["baseline_index"] = previous["index"]

    metric_passes, query_passes, significance = {}, {}, {}

    for key, minimum, metrics in (
        ("relevant_or_partial", 1, cfg.get("metrics", METRICS)),
        ("relevant_only", 2, cfg.get("strict_metrics", STRICT)),
    ):
        if not metrics:
            continue

        qrels = qrels_for(name, level, set(query_ids), minimum)
        means, per_method = {}, {}

        for run in runs:
            values = evaluate(
                qrels, run, metrics,
                make_comparable=True,
                save_results_in_run=True
            )
            if not isinstance(values, dict):
                values = {metrics[0]: values}

            means[run.name] = {
                m: float(values[m]) for m in metrics
            }
            per_method[run.name] = {
                m: {
                    qid: float(score)
                    for qid, score in dict(run.scores[m]).items()
                }
                for m in metrics
            }
            print(
                f"  {name}/{level}/{key} {run.name}: {means[run.name]}",
                flush=True
            )

        per_query = {
            qid: {
                method: {
                    m: scores[qid]
                    for m, scores in metrics_by_name.items()
                    if qid in scores
                }
                for method, metrics_by_name in per_method.items()
                if any(qid in scores for scores in metrics_by_name.values())
            }
            for qid in query_ids
        }
        per_query = {
            qid: result for qid, result in per_query.items() if result
        }

        metric_passes[key] = {
            "relevance_level": minimum,
            "metrics": means
        }
        query_passes[key] = {
            "relevance_level": minimum,
            "queries": per_query
        }

        if len(runs) >= 2:
            test = cfg.get("stat_test", "student")
            p = float(cfg.get("max_p", 0.05))
            report = compare(
                qrels, runs, metrics,
                stat_test=test, max_p=p,
                make_comparable=True
            )
            detail = report.to_dict()
            comparisons = {}

            for a, b in combinations([r.name for r in runs], 2):
                pvals = detail[a]["comparisons"][b]
                counts = detail[a]["win_tie_loss"][b]

                comparisons[f"{a} vs {b}"] = {
                    m: {
                        "p": float(f"{pvals[m]:.3g}"),
                        "better": (
                            a if means[a][m] > means[b][m] else b
                        ) if pvals[m] < p else None,
                        "win_tie_loss": dict(counts[m])
                    }
                    for m in metrics
                }

            significance[key] = {
                "relevance_level": minimum,
                "comparisons": comparisons
            }

    prefix = f"{name}_{level}{'' if stem else '_no-stem'}"
    summary = {
        **context,
        "performance": performance,
        "trec_runs": paths,
        "relevance_passes": metric_passes,
        "queries_file": str(
            (OUT / f"{prefix}_queries.json").relative_to(ROOT)
        )
    }

    if significance:
        summary["significance_file"] = str(
            (OUT / f"{prefix}_significance.json").relative_to(ROOT)
        )

    write(OUT / f"{prefix}.json", summary)
    write(
        OUT / f"{prefix}_queries.json",
        {**context, "relevance_passes": query_passes}
    )

    if significance:
        write(
            OUT / f"{prefix}_significance.json",
            {
                **context,
                "stat_test": cfg.get("stat_test", "student"),
                "max_p": float(cfg.get("max_p", 0.05)),
                "relevance_passes": significance,
            }
        )

    print(f"Saved metrics: {OUT / f'{prefix}.json'}", flush=True)


def main(config=DEFAULT):
    cfg = yaml.safe_load(path(config).read_text())

    if not cfg.get("models"):
        raise ValueError("Specify models in config['models']")

    jobs = cfg.get("sets", [])
    if not jobs or any(
        job.get("name") not in ("train", "study", "test")
        or not isinstance(job.get("metrics"), bool)
        for job in jobs
    ):
        raise ValueError("sets must specify name and metrics: true/false")

    levels = cfg.get("levels", ["paper", "paragraph"])
    if not levels or any(
        level not in ("paper", "paragraph") for level in levels
    ):
        raise ValueError("levels must contain paper and/or paragraph")

    k = int(cfg.get("k", 100))
    if k < 1:
        raise ValueError("k must be positive")

    query_sets = {
        job["name"]: queries_for(job["name"], cfg.get("limit"))
        for job in jobs
    }

    for job in jobs:
        if not query_sets[job["name"]]:
            raise ValueError(f"No queries in {job['name']}")
        if job["metrics"]:
            for level in levels:
                filename = (
                    DATA / "qrels"
                    / f"{job['name'].capitalize()}_{level}_qrel.tsv"
                )
                if not filename.is_file():
                    raise FileNotFoundError(f"Missing QREL file: {filename}")

    generated = {}

    for label, spec in cfg["models"].items():
        if spec["type"] not in ("sparse", "dense"):
            raise ValueError(f"{label}: type must be sparse or dense")

        checkpoint = path(spec["checkpoint"])
        index = path(spec["index_path"])

        state = json.loads((index / "checkpoint.json").read_text())
        corpus_count = sum(
            1 for _ in (DATA / "JSON Files").glob("*.json")
        )
        if state["count"] != corpus_count:
            raise RuntimeError(
                f"{label}: incomplete index ({state['count']}/{corpus_count})"
            )

        cls = (
            SparseEncoder
            if spec["type"] == "sparse"
            else SentenceTransformer
        )
        model = cls(str(checkpoint))
        model.eval()
        print(
            f"Loaded {label} ({spec['type']}): {checkpoint}",
            flush=True
        )

        for job in jobs:
            name = job["name"]
            queries = query_sets[name]
            answers = retrieve(
                model, spec, index, queries, levels, k
            )

            for level, (ranked, latency, info) in answers.items():
                filename = run_path(
                    cfg["team_name"], level, label, name
                )
                filename.parent.mkdir(parents=True, exist_ok=True)

                Run.from_dict(
                    ranked, name=filename.stem
                ).save(str(filename), kind="trec")

                generated[(name, level, label)] = (
                    ranked, latency, info
                )
                print(
                    f"Saved run: {filename.relative_to(ROOT)}",
                    flush=True
                )

        del model

    for job in jobs:
        for level in levels:
            evaluate_level(
                cfg, job, level,
                query_sets[job["name"]], generated
            )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)