"""
Evaluate BM25 vs TF-IDF against the course qrels using ranx.

Settings live in config.yaml next to this module. Writes TREC run files under
data/runs/, then prints and saves metric scores.

Run from the repository root:

    python -m apps.evaluation.evaluate
    python -m apps.evaluation.evaluate apps/evaluation/config.yaml
"""

from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path

import yaml
from ranx import Qrels, Run, compare, evaluate

from apps.retrieval import bm25, tfidf

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.yaml"

DATA_DIR = ROOT / "data"
QRELS_DIR = DATA_DIR / "qrels"
RUNS_DIR = DATA_DIR / "runs"
METRICS_DIR = DATA_DIR / "evaluation"

QUERY_FILES = {
    "study": DATA_DIR / "Study.json",
    "test": DATA_DIR / "Test.json",
}

METHODS = {
    "bm25": bm25.retrieve,
    "tfidf": tfidf.retrieve,
}

STAT_TESTS = ("student", "fisher")


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    try:
        team_name = str(cfg["team_name"])
    except KeyError as err:
        raise KeyError(f"'team_name' is missing from {path}") from err

    set_name = str(cfg.get("set", "study")).lower()
    if set_name not in QUERY_FILES:
        raise ValueError(f"set must be one of {sorted(QUERY_FILES)}, got {set_name!r}")

    levels = cfg.get("levels") or ["paper", "paragraph"]
    if isinstance(levels, str):
        levels = [levels]
    levels = [str(level).lower() for level in levels]
    for level in levels:
        if level not in ("paper", "paragraph"):
            raise ValueError(f"Unknown level {level!r} (expected paper or paragraph)")

    methods = cfg.get("methods") or ["bm25", "tfidf"]
    if isinstance(methods, str):
        methods = [methods]
    methods = [str(m).lower() for m in methods]
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r} (expected {sorted(METHODS)})")

    k = int(cfg.get("k", 100))
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    stem = bool(cfg.get("stem", True))

    limit = cfg.get("limit", None)
    if limit is not None:
        limit = int(limit)
        if limit < 1:
            raise ValueError(f"limit must be >= 1 or null, got {limit}")

    metrics = cfg.get("metrics") or [
        "ndcg",
        "ndcg@5",
        "mrr",
        "precision@5",
        "precision@10",
    ]
    if isinstance(metrics, str):
        metrics = [metrics]
    metrics = [str(m) for m in metrics]
    if not metrics:
        raise ValueError("metrics list must not be empty")

    strict_metrics = cfg.get("strict_metrics") or []
    if isinstance(strict_metrics, str):
        strict_metrics = [strict_metrics]
    strict_metrics = [str(m) for m in strict_metrics]

    stat_test = str(cfg.get("stat_test", "student")).lower()
    if stat_test not in STAT_TESTS:
        raise ValueError(f"stat_test must be one of {list(STAT_TESTS)}, got {stat_test!r}")

    max_p = float(cfg.get("max_p", 0.05))
    if not 0 < max_p < 1:
        raise ValueError(f"max_p must be between 0 and 1, got {max_p}")

    return {
        "team_name": team_name,
        "set": set_name,
        "levels": levels,
        "methods": methods,
        "k": k,
        "stem": stem,
        "limit": limit,
        "metrics": metrics,
        "strict_metrics": strict_metrics,
        "stat_test": stat_test,
        "max_p": max_p,
    }


def load_queries(set_name: str, limit: int | None = None) -> list[tuple[str, str]]:
    path = QUERY_FILES[set_name]
    if not path.exists():
        raise FileNotFoundError(f"Query file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        rows = json.load(fh)
    queries = [(str(row["query_id"]), row["query"]) for row in rows]
    if limit is not None:
        queries = queries[:limit]
    return queries


def load_qrels(
    set_name: str,
    level: str,
    query_ids: set[str],
    rel_lvl: int,
) -> Qrels:
    """Qrels for one level, limited to `query_ids`, at the given relevance level.

    rel_lvl=1 counts Relevant and Partially Relevant; rel_lvl=2 counts only the
    Relevant (grade 2) judgments.
    """
    name = f"{set_name.capitalize()}_{level}_qrel.tsv"
    path = QRELS_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"No qrels for set={set_name!r} level={level!r} at {path}"
        )
    judged = Qrels.from_file(str(path), kind="trec").to_dict()
    qrels = Qrels.from_dict(
        {qid: docs for qid, docs in judged.items() if qid in query_ids}
    )
    qrels.set_relevance_level(rel_lvl=rel_lvl)
    return qrels


def run_tag(team: str, method: str, level: str, set_name: str, stem: bool) -> str:
    """Submission run tag, e.g. TheSearchParty_paper_bm25_study.

    Stemming is the default, so only unstemmed ablations get a suffix; that
    keeps them from overwriting the files we hand in.
    """
    tag = f"{team}_{level}_{method}_{set_name}"
    return tag if stem else f"{tag}_no-stem"


def trec_path(team: str, method: str, level: str, set_name: str, stem: bool) -> Path:
    return RUNS_DIR / f"{run_tag(team, method, level, set_name, stem)}.tsv"


def output_stem(level: str, stem: bool) -> str:
    return level if stem else f"{level}_no-stem"


def metrics_path(level: str, stem: bool) -> Path:
    return METRICS_DIR / f"{output_stem(level, stem)}.json"


def significance_path(level: str, stem: bool) -> Path:
    return METRICS_DIR / f"{output_stem(level, stem)}_significance.json"


def build_run(
    team: str,
    method: str,
    queries: list[tuple[str, str]],
    level: str,
    set_name: str,
    stem: bool,
    k: int,
) -> Run:
    retrieve = METHODS[method]
    ranked: dict[str, dict[str, float]] = {}
    total = len(queries)
    started = time.perf_counter()

    for i, (qid, text) in enumerate(queries, start=1):
        hits = retrieve(text, level, stem, k)
        ranked[qid] = {doc_id: float(score) for doc_id, score in hits.items()}
        if i == 1 or i % 25 == 0 or i == total:
            elapsed = time.perf_counter() - started
            print(f"  [{method}/{level}] {i}/{total} queries ({elapsed:.1f}s)", flush=True)

    return Run.from_dict(ranked, name=run_tag(team, method, level, set_name, stem))


def format_metrics(scores: dict) -> str:
    parts = [f"{key}={float(val):.4f}" for key, val in scores.items()]
    return "  ".join(parts)


def score_pass(
    qrels: Qrels,
    runs: list[Run],
    metrics: list[str],
    label: str,
) -> dict[str, dict]:
    """Scores every run against one relevance level."""
    print(f"\n-- {label} --", flush=True)

    per_method = {}
    for run in runs:
        scores = evaluate(qrels, run, metrics, make_comparable=True)
        print(f"  {run.name}: {format_metrics(scores)}", flush=True)
        per_method[run.name] = {m: float(scores[m]) for m in metrics}

    return per_method


def compare_pass(
    qrels: Qrels,
    runs: list[Run],
    metrics: list[str],
    scores: dict[str, dict],
    stat_test: str,
    max_p: float,
) -> dict:
    """Pairwise significance results, keyed "<a> vs <b>" then by metric.

    Win/tie/loss counts are per query, from the perspective of the first method
    in the pair. These land in their own file, so keeping them costs nothing in
    the metrics summary.
    """
    report = compare(
        qrels,
        runs,
        metrics,
        max_p=max_p,
        stat_test=stat_test,
        make_comparable=True,
    )
    print(report)

    detail = report.to_dict()
    results: dict[str, dict] = {}
    for a, b in combinations([run.name for run in runs], 2):
        p_values = detail[a]["comparisons"][b]
        counts = detail[a]["win_tie_loss"][b]
        results[f"{a} vs {b}"] = {
            metric: {
                # 3 significant digits keeps very small p-values readable.
                "p": float(f"{p_values[metric]:.3g}"),
                "better": (
                    (a if scores[a][metric] > scores[b][metric] else b)
                    if p_values[metric] < max_p
                    else None
                ),
                "win_tie_loss": dict(counts[metric]),
            }
            for metric in metrics
        }
    return results


def evaluate_level(
    team: str,
    set_name: str,
    level: str,
    queries: list[tuple[str, str]],
    methods: list[str],
    stem: bool,
    k: int,
    metrics: list[str],
    strict_metrics: list[str],
    stat_test: str,
    max_p: float,
) -> dict:
    print(f"\n=== {set_name} / {level} (stem={stem}, k={k}) ===", flush=True)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Retrieval is the slow part, so each run is built once and scored twice.
    runs: list[Run] = []
    for method in methods:
        print(f"Retrieving with {method}…", flush=True)
        run = build_run(team, method, queries, level, set_name, stem, k)
        path = trec_path(team, method, level, set_name, stem)
        run.save(str(path), kind="trec")
        print(f"  wrote TREC run → {path.relative_to(ROOT)}", flush=True)

        # The saved file keeps the full run tag; the score tables below are keyed
        # on run.name, which reads better as just the method.
        run.name = method
        runs.append(run)

    query_ids = {qid for qid, _ in queries}
    metric_passes: dict[str, dict] = {}
    significance_passes: dict[str, dict] = {}

    for key, rel_lvl, metric_names, label in (
        ("relevant_or_partial", 1, metrics, "Relevant + Partially Relevant"),
        ("relevant_only", 2, strict_metrics, "Relevant only (grade 2)"),
    ):
        if not metric_names:
            continue
        qrels = load_qrels(set_name, level, query_ids, rel_lvl)
        scores = score_pass(qrels, runs, metric_names, label)
        metric_passes[key] = {"relevance_level": rel_lvl, "metrics": scores}

        if len(runs) >= 2:
            print(f"\n  significance (paired {stat_test} t-test, p < {max_p}):", flush=True)
            significance_passes[key] = {
                "relevance_level": rel_lvl,
                "comparisons": compare_pass(
                    qrels, runs, metric_names, scores, stat_test, max_p
                ),
            }

    context = {"set": set_name, "level": level, "stem": stem, "k": k}

    # Scores and significance go to separate files: the metrics summary is what
    # gets read constantly, so it stays short.
    summary = {
        **context,
        "relevance_passes": metric_passes,
        "trec_runs": {
            method: str(trec_path(team, method, level, set_name, stem).relative_to(ROOT))
            for method in methods
        },
    }
    if significance_passes:
        summary["significance_file"] = str(
            significance_path(level, stem).relative_to(ROOT)
        )

    out = metrics_path(level, stem)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved metric summary → {out.relative_to(ROOT)}", flush=True)

    if significance_passes:
        report = {
            **context,
            "stat_test": stat_test,
            "max_p": max_p,
            "relevance_passes": significance_passes,
        }
        out = significance_path(level, stem)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved significance report → {out.relative_to(ROOT)}", flush=True)

    return summary


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    config_path = Path(argv[0]) if argv else DEFAULT_CONFIG
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()

    cfg = load_config(config_path)
    print(f"Loaded config from {config_path.relative_to(ROOT) if ROOT in config_path.parents else config_path}")

    if cfg["set"] != "study":
        qrel_probe = QRELS_DIR / f"{cfg['set'].capitalize()}_paper_qrel.tsv"
        if not qrel_probe.exists():
            print(
                f"No qrels under {QRELS_DIR} for set={cfg['set']!r}. "
                "Only the Study set is currently labeled.",
                file=sys.stderr,
            )
            return 1

    queries = load_queries(cfg["set"], cfg["limit"])
    print(f"Loaded {len(queries)} queries from {QUERY_FILES[cfg['set']].name}")

    for level in cfg["levels"]:
        evaluate_level(
            team=cfg["team_name"],
            set_name=cfg["set"],
            level=level,
            queries=queries,
            methods=cfg["methods"],
            stem=cfg["stem"],
            k=cfg["k"],
            metrics=cfg["metrics"],
            strict_metrics=cfg["strict_metrics"],
            stat_test=cfg["stat_test"],
            max_p=cfg["max_p"],
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
