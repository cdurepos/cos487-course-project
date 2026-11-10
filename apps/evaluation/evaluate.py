"""
Evaluate neural retrieval; save MaxP runs and compare pooling in memory.

Usage:
    python -m apps.evaluation.evaluate [config]
"""

import json
import sys
import time
from itertools import combinations
from pathlib import Path

import yaml
from ranx import Qrels, Run, compare, evaluate
from sentence_transformers import SparseEncoder

from apps.retrieval.dense import search as dense_search
from apps.retrieval.sparse import search as sparse_search, warmup
from apps.utils.sparse import catalog

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RUNS = DATA / "runs"
OUT = DATA / "evaluation"
METRICS = ["ndcg", "ndcg@5", "mrr", "precision@5", "precision@10"]


def evaluate_level(cfg, job, level, queries, generated):
    name, team = job["name"], cfg["team_name"]
    stem = cfg.get("stem", True)
    baseline = cfg.get("baseline_methods", [])
    candidates = generated[name][level]
    runs, paths = {}, {}

    # Load Part 1 baselines.
    for method in baseline:
        suffix = "_no-stem" if method in ("bm25", "tfidf") and not stem else ""
        file = RUNS / f"{team}_{level}_{method}_{name}{suffix}.tsv"
        if not file.is_file():
            raise FileNotFoundError(file)
        runs[method] = Run.from_file(str(file), kind="trec", name=method)
        paths[method] = str(file.relative_to(ROOT))

    # Include pooling in evaluation, but not in saved run paths.
    for method, (run, _, _, experimental) in candidates.items():
        runs[method] = run
        spec = cfg["models"].get(method, {})
        save_id = spec.get("save_id")
        if not experimental and save_id:
            paths[method] = str(
                (RUNS / f"{team}_{level}_{save_id}_{name}.tsv").relative_to(ROOT)
            )

    prefix = f"{name}_{level}{'_no-stem' if not stem else ''}"
    prior = OUT / f"{prefix}.json"
    old = json.loads(prior.read_text()).get("performance", {}) if prior.exists() else {}

    performance = {
        "indexes": {m: v[2] for m, v in candidates.items()},
        "latency_ms": {m: v[1] for m, v in candidates.items()},
    }
    performance["latency_ms"].update({
        m: old["latency_ms"][m] for m in baseline
        if m in old.get("latency_ms", {})
    })
    if "baseline_index" in old or "index" in old:
        performance["baseline_index"] = old.get("baseline_index", old.get("index"))

    metric_passes, query_passes, significance = {}, {}, {}

    for key, minimum, metrics in (
        ("relevant_or_partial", 1, cfg.get("metrics", METRICS)),
        ("relevant_only", 2, cfg.get("strict_metrics", ["mrr", "precision@5"])),
    ):
        if not metrics:
            continue

        file = DATA / "qrels" / f"{name.capitalize()}_{level}_qrel.tsv"
        judged = Qrels.from_file(str(file), kind="trec").to_dict()
        qrels = Qrels.from_dict({q: judged[q] for q, _ in queries if q in judged})
        qrels.set_relevance_level(rel_lvl=minimum)

        means = {}
        detailed = {qid: {} for qid, _ in queries}

        for method, run in runs.items():
            values = evaluate(
                qrels, run, metrics,
                make_comparable=True, save_results_in_run=True,
            )
            means[method] = {m: float(values[m]) for m in metrics}

            # Only primary runs receive detailed query-level analysis.
            if method not in candidates or (
                not candidates[method][3]
                and cfg["models"].get(method, {}).get("save_runs", False)
            ):
                for m in metrics:
                    for qid, score in run.scores[m].items():
                        if qid in detailed:
                            detailed[qid].setdefault(method, {})[m] = float(score)

            print(f"  {name}/{level}/{key} {method}: {means[method]}", flush=True)

        metric_passes[key] = {"relevance_level": minimum, "metrics": means}
        query_passes[key] = {
            "relevance_level": minimum,
            "queries": {q: v for q, v in detailed.items() if v},
        }

        # Pairwise significance testing includes pooling.
        if len(runs) > 1:
            p = float(cfg.get("max_p", 0.05))
            report = compare(
                qrels, list(runs.values()), metrics,
                stat_test=cfg.get("stat_test", "student"),
                max_p=p, make_comparable=True,
            ).to_dict()

            comparisons = {}
            for a, b in combinations(runs, 2):
                pvals = report[a]["comparisons"][b]
                counts = report[a]["win_tie_loss"][b]

                comparisons[f"{a} vs {b}"] = {
                    m: {
                        "p": float(f"{pvals[m]:.3g}"),
                        "better": (
                            a if means[a][m] > means[b][m] else b
                        ) if pvals[m] < p else None,
                        "win_tie_loss": dict(counts[m]),
                    }
                    for m in metrics
                }

            significance[key] = {
                "relevance_level": minimum,
                "comparisons": comparisons,
            }

    context = {
        "set": name, "level": level,
        "stem": stem, "k": cfg.get("k", 100),
    }

    summary = {
        **context,
        "performance": performance,
        "trec_runs": paths,
        "experimental_methods": [m for m, v in candidates.items() if v[3]],
        "relevance_passes": metric_passes,
        "queries_file": str((OUT / f"{prefix}_queries.json").relative_to(ROOT)),
    }

    if significance:
        summary["significance_file"] = str(
            (OUT / f"{prefix}_significance.json").relative_to(ROOT)
        )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{prefix}.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / f"{prefix}_queries.json").write_text(
        json.dumps({**context, "relevance_passes": query_passes}, indent=2) + "\n"
    )
    if significance:
        (OUT / f"{prefix}_significance.json").write_text(json.dumps({
            **context,
            "stat_test": cfg.get("stat_test", "student"),
            "max_p": cfg.get("max_p", 0.05),
            "relevance_passes": significance,
        }, indent=2) + "\n")

    print(f"Saved metrics: {OUT / f'{prefix}.json'}", flush=True)


def main(config=ROOT / "configs" / "eval.yaml"):
    file = Path(config).expanduser()
    cfg = yaml.safe_load((file if file.is_absolute() else ROOT / file).read_text())
    if not cfg.get("models") or not cfg.get("sets"):
        raise ValueError("Configure models and query sets")

    levels, k = cfg.get("levels", ["paper", "paragraph"]), int(cfg.get("k", 100))
    if k < 1 or any(level not in ("paper", "paragraph") for level in levels):
        raise ValueError("Invalid levels or k")

    sets = {}
    for job in cfg["sets"]:
        name = job["name"]
        rows = json.loads((DATA / f"{name.capitalize()}.json").read_text())
        if cfg.get("limit"):
            rows = rows[:cfg["limit"]]
        sets[name] = [(str(r["query_id"]), r["query"]) for r in rows]

        if job["metrics"]:
            for level in levels:
                file = DATA / "qrels" / f"{name.capitalize()}_{level}_qrel.tsv"
                if not file.exists():
                    raise FileNotFoundError(file)

    generated = {name: {level: {} for level in levels} for name in sets}
    corpus_count = sum(1 for _ in (DATA / "JSON Files").glob("*.json"))
    sparse = catalog()

    for label, spec in cfg["models"].items():
        kind = spec.get("type", "sparse")
        model = index = None
        if kind == "sparse":
            if label not in sparse:
                raise KeyError(f"{label} is missing from configs/sparse.yaml")
            checkpoint, index = sparse[label]["checkpoint"], sparse[label]["index"]
            count = json.loads((index / "checkpoint.json").read_text())["count"]
            if count != corpus_count:
                raise RuntimeError(f"{label}: incomplete index ({count}/{corpus_count})")
            model = SparseEncoder(str(checkpoint))
            model.eval()
            print(f"Loaded {label}: {checkpoint}", flush=True)
        elif kind != "dense":
            raise ValueError(f"{label}: unknown model type {kind!r}")

        for job in cfg["sets"]:
            name = job["name"]
            if name == "test" and not spec.get("do_test", False):
                continue
            queries = sets[name]
            if kind == "sparse":
                calls = [(level, "maxp", False) for level in levels]
                if job["metrics"] and "paper" in levels:
                    calls.append(("paper", "pool", True))
            else:
                level = spec.get("level")
                if level not in ("paper", "paragraph"):
                    raise ValueError(f"{label}: dense models must specify level as 'paper' or 'paragraph'")
                if level not in levels:
                    raise ValueError(f"{label}: configured level {level!r} is not included in evaluation levels {levels}")
                calls = [(level, None, False)]

            for level, strategy, experimental in calls:
                if kind == "sparse":
                    warmup(index, level, strategy)
                    ranked = {}
                    latency = {}
                    for qid, text in queries:
                        start = time.perf_counter()
                        ranked[qid] = sparse_search(model, level, text, k, strategy)
                        latency[qid] = (time.perf_counter() - start) * 1000
                    source = (
                        "papers" if level == "paper" and strategy == "pool"
                        else "paragraphs"
                    )
                    files = [
                        *index.glob(f"{source}_*.npz"),
                        *index.glob(f"{source}_*_ids.json"),
                    ]
                    info = {
                        "path": (
                            str(index.relative_to(ROOT))
                            if index.is_relative_to(ROOT) else str(index)
                        ),
                        "source": source,
                        "strategy": strategy if level == "paper" else "paragraph",
                        "bytes": sum(path.stat().st_size for path in files),
                        "files": len(files),
                    }
                elif kind == "dense":
                    ranked = {}
                    latency = {}
                    info = {}
                    for qid, text in queries:
                        start = time.perf_counter()
                        results = dense_search(
                            query=text,
                            training=spec["training"],
                            level=level,
                            paper_method=spec.get("paper_method"),
                            paragraph_method=spec.get("paragraph_method"),
                            crossencoder_training=spec.get("crossencoder_training"),
                            k=k,
                        )
                        latency[qid] = (time.perf_counter() - start) * 1000
                        ranked[qid] = dict(results)

                method = f"{label}-pool" if experimental else label
                run = Run.from_dict(ranked, name=method)
                generated[name][level][method] = (run, latency, info, experimental)

                save_id = spec.get("save_id")
                if not experimental and save_id:
                    file = RUNS / f"{cfg['team_name']}_{level}_{save_id}_{name}.tsv"
                    file.parent.mkdir(parents=True, exist_ok=True)
                    run.save(str(file), kind="trec")
                    print(f"Saved run: {file.relative_to(ROOT)}", flush=True)
        del model

    for job in cfg["sets"]:
        if job["metrics"]:
            for level in levels:
                evaluate_level(cfg, job, level, sets[job["name"]], generated)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "configs" / "eval.yaml")