"""
Benchmark harness for the scoring path. Run:  python -m flowsentry.bench

Measures, on real flows from the committed BCCC sample, with a trained artifact:

  * single_row: the serving path as /predict sees it (feature dict -> row ->
    impute -> sequential two-stage predict via FlowScorer.score_one), timed per
    call, reported as mean/p50/p95/p99 and the implied sequential flows/s. The
    p95/p99 tail is real: it contains the flows that escalate to Stage 2.
  * single_row_native_pool: the same work through the forests' native
    predict_proba, which re-spawns a joblib thread pool per call. This is the
    path the service used before the sequential fix; it is kept in the benchmark
    so the before/after speedup stays reproducible, not folklore.
  * batch: FlowScorer.score_batch on a whole matrix at several sizes, flows/s
    (median of several repetitions).

Every number is measured on the machine running the benchmark and written to
artifacts/benchmark.json together with the library versions and CPU count, so a
quoted figure can always be traced to an environment. Nothing here is estimated.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .scoring import ARTIFACT_DIR, FlowScorer, load_bundle
from .stream import load_stream

BATCH_SIZES = [1024, 8192, 0]  # 0 = the whole committed sample
BATCH_REPS = 5
NATIVE_POOL_ROWS = 100  # the slow path gets fewer reps; it costs ~45 ms per call


def _percentiles(latencies_ms: np.ndarray) -> dict:
    return {
        "mean_ms": round(float(latencies_ms.mean()), 3),
        "p50_ms": round(float(np.percentile(latencies_ms, 50)), 3),
        "p95_ms": round(float(np.percentile(latencies_ms, 95)), 3),
        "p99_ms": round(float(np.percentile(latencies_ms, 99)), 3),
        "implied_flows_per_s": round(1000.0 / float(latencies_ms.mean()), 1),
        "n_calls": int(latencies_ms.size),
    }


def _feature_dicts(X: np.ndarray, feature_names: list[str], idx: np.ndarray) -> list[dict]:
    """Feature dicts as a real /predict request would carry them (NaN entries
    dropped: a client simply omits features it does not have)."""
    dicts = []
    for i in idx:
        row = X[i]
        dicts.append(
            {
                name: float(v)
                for name, v in zip(feature_names, row, strict=True)
                if not np.isnan(v)
            }
        )
    return dicts


def bench_single_row(scorer: FlowScorer, X: np.ndarray, rows: int, seed: int) -> dict:
    """Time FlowScorer.score_one per call, including the dict -> row build,
    so the percentiles are honest per-request numbers."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(X.shape[0], size=min(rows, X.shape[0]), replace=False)
    payloads = _feature_dicts(X, scorer.feature_names, idx)

    for p in payloads[:3]:  # warmup
        scorer.score_one(p)
    lat = np.empty(len(payloads), dtype=float)
    for k, p in enumerate(payloads):
        t0 = time.perf_counter()
        scorer.score_one(p)
        lat[k] = (time.perf_counter() - t0) * 1000.0
    return _percentiles(lat)


def bench_single_row_native_pool(
    scorer: FlowScorer, X: np.ndarray, rows: int, seed: int
) -> dict:
    """The pre-fix serving path: per-row imputer.transform + predict with the
    native thread-pool predict_proba. Kept so the speedup stays measurable."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(X.shape[0], size=min(rows, X.shape[0]), replace=False)
    imputer, model = scorer.imputer, scorer.model

    for i in idx[:3]:  # warmup
        model.predict_detail(imputer.transform(X[i : i + 1]), reject_threshold=0.0)
    lat = np.empty(idx.size, dtype=float)
    for k, i in enumerate(idx):
        row = X[i : i + 1]
        t0 = time.perf_counter()
        Xi = imputer.transform(row)
        model.predict_detail(Xi, reject_threshold=0.0)
        lat[k] = (time.perf_counter() - t0) * 1000.0
    return _percentiles(lat)


def bench_batch(
    scorer: FlowScorer, X: np.ndarray, sizes: list[int], reps: int, seed: int
) -> list[dict]:
    """Throughput of FlowScorer.score_batch per batch size, median of `reps` runs."""
    rng = np.random.RandomState(seed)
    out = []
    for size in sizes:
        n = X.shape[0] if size <= 0 else min(size, X.shape[0])
        idx = rng.choice(X.shape[0], size=n, replace=False)
        Xb = X[idx]
        walls = []
        for _ in range(reps):
            t0 = time.perf_counter()
            scorer.score_batch(Xb, reject_threshold=0.0)
            walls.append(time.perf_counter() - t0)
        wall = float(np.median(walls))
        out.append(
            {
                "batch_size": n,
                "median_wall_s": round(wall, 4),
                "flows_per_s": round(n / wall, 1),
                "reps": reps,
            }
        )
    return out


def environment() -> dict:
    import joblib
    import pandas
    import sklearn

    return {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.system(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def run(rows: int, seed: int, out_path: Path) -> dict:
    scorer = FlowScorer.from_bundle(load_bundle())
    X, _ = load_stream(0)
    print(f"[bench] {X.shape[0]} flows loaded; timing single-row ({rows} calls) ...")
    single = bench_single_row(scorer, X, rows, seed)
    print(
        f"[bench] single-row (serving path)  mean={single['mean_ms']} ms  "
        f"p50={single['p50_ms']} ms  p95={single['p95_ms']} ms  p99={single['p99_ms']} ms  "
        f"-> {single['implied_flows_per_s']} flows/s sequential"
    )
    native = bench_single_row_native_pool(scorer, X, NATIVE_POOL_ROWS, seed)
    print(
        f"[bench] single-row (native pool, pre-fix path)  mean={native['mean_ms']} ms  "
        f"p50={native['p50_ms']} ms  p95={native['p95_ms']} ms  "
        f"-> {native['implied_flows_per_s']} flows/s sequential"
    )
    print("[bench] timing batch throughput ...")
    batches = bench_batch(scorer, X, BATCH_SIZES, BATCH_REPS, seed)
    for b in batches:
        print(
            f"[bench] batch {b['batch_size']:>6} rows: {b['flows_per_s']:>10,.0f} flows/s "
            f"(median of {b['reps']} reps, {b['median_wall_s']}s wall)"
        )
    report = {
        "what": (
            "measured scoring-path benchmark; single_row is the per-request serving "
            "path (FlowScorer.score_one), single_row_native_pool is the pre-fix path "
            "kept for comparison, batch is FlowScorer.score_batch"
        ),
        "environment": environment(),
        "single_row": single,
        "single_row_native_pool": native,
        "batch": batches,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[bench] wrote {out_path}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark the FlowSentry scoring path.")
    ap.add_argument("--rows", type=int, default=300, help="single-row calls to time")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--json", type=Path, default=ARTIFACT_DIR / "benchmark.json",
        help="where to write the JSON report",
    )
    args = ap.parse_args()
    run(args.rows, args.seed, args.json)


if __name__ == "__main__":
    main()
