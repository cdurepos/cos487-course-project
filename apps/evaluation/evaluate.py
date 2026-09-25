"""
Evaluate BM25 vs TF-IDF against the course qrels using ranx.

Settings live in config.yaml next to this module. Writes TREC run files under
data/runs/, then prints metric scores and a head-to-head comparison.

Run from the repository root:

    python -m apps.evaluation.evaluate
    python -m apps.evaluation.evaluate apps/evaluation/config.yaml
"""

from __future__ import annotations

import json
import sys
import time
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


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

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

    metrics = cfg.get("metrics") or ["map", "mrr", "ndcg@10", "recall@100"]
    if isinstance(metrics, str):
        metrics = [metrics]
    metrics = [str(m) for m in metrics]
    if not metrics:
        raise ValueError("metrics list must not be empty")

    return {
        "set": set_name,
        "levels": levels,
        "methods": methods,
        "k": k,
        "stem": stem,
        "limit": limit,
        "metrics": metrics,
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


def load_qrels(set_name: str, level: str) -> Qrels:
    name = f"{set_name.capitalize()}_{level}_qrel.tsv"
    path = QRELS_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"No qrels for set={set_name!r} level={level!r} at {path}"
        )
    return Qrels.from_file(str(path), kind="trec")


def run_name(level: str, stem: bool) -> str:
    """Short run tag and filename stem, e.g. paper_stem or paragraph_no-stem."""
    return f"{level}_stem" if stem else f"{level}_no-stem"


def trec_path(method: str, level: str, stem: bool) -> Path:
    """e.g. data/runs/bm25/paper_stem.tsv or data/runs/tfidf/paragraph_no-stem.tsv."""
    return RUNS_DIR / method / f"{run_name(level, stem)}.tsv"


def build_run(
    method: str,
    queries: list[tuple[str, str]],
    level: str,
    stem: bool,
    k: int,
) -> Run:
    retrieve = METHODS[method]
    name = f"{method}_{run_name(level, stem)}"
    ranked: dict[str, dict[str, float]] = {}
    total = len(queries)
    started = time.perf_counter()

    for i, (qid, text) in enumerate(queries, start=1):
        hits = retrieve(text, level, stem, k)
        ranked[qid] = {doc_id: float(score) for doc_id, score in hits.items()}
        if i == 1 or i % 25 == 0 or i == total:
            elapsed = time.perf_counter() - started
            print(f"  [{method}/{level}] {i}/{total} queries ({elapsed:.1f}s)", flush=True)

    return Run.from_dict(ranked, name=name)


def format_metrics(scores: dict) -> str:
    parts = [f"{key}={float(val):.4f}" for key, val in scores.items()]
    return "  ".join(parts)


def evaluate_level(
    set_name: str,
    level: str,
    queries: list[tuple[str, str]],
    methods: list[str],
    stem: bool,
    k: int,
    metrics: list[str],
) -> dict:
    print(f"\n=== {set_name} / {level} (stem={stem}, k={k}) ===", flush=True)
    qrels = load_qrels(set_name, level)

    # When limit is set, score only the queries we actually ran.
    query_ids = {qid for qid, _ in queries}
    qrels_all = qrels.to_dict()
    qrels = Qrels.from_dict(
        {qid: docs for qid, docs in qrels_all.items() if qid in query_ids}
    )

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    runs: list[Run] = []
    per_method: dict[str, dict] = {}

    for method in methods:
        print(f"Retrieving with {method}…", flush=True)
        run = build_run(method, queries, level, stem, k)
        path = trec_path(method, level, stem)
        path.parent.mkdir(parents=True, exist_ok=True)
        run.save(str(path), kind="trec")
        print(f"  wrote TREC run → {path.relative_to(ROOT)}", flush=True)

        scores = evaluate(qrels, run, metrics, make_comparable=True)
        print(f"  {method}: {format_metrics(scores)}", flush=True)
        per_method[method] = {m: float(scores[m]) for m in metrics}
        runs.append(run)

    if len(runs) >= 2:
        print("\nHead-to-head (ranx.compare, paired Student t-test):", flush=True)
        report = compare(
            qrels,
            runs,
            metrics,
            max_p=0.01,
            stat_test="student",
            make_comparable=True,
        )
        print(report)

    summary = {
        "set": set_name,
        "level": level,
        "stem": stem,
        "k": k,
        "metrics": per_method,
        "trec_runs": {
            method: str(trec_path(method, level, stem).relative_to(ROOT))
            for method in methods
        },
    }
    out = METRICS_DIR / f"{run_name(level, stem)}.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved metric summary → {out.relative_to(ROOT)}", flush=True)
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
            set_name=cfg["set"],
            level=level,
            queries=queries,
            methods=cfg["methods"],
            stem=cfg["stem"],
            k=cfg["k"],
            metrics=cfg["metrics"],
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
