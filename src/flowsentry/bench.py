"""
Benchmark harness for the scoring path. Run:  python -m flowsentry.bench

Measures, on real flows from the committed BCCC sample, with a trained artifact:

  * single-row latency: the serving path as /predict sees it (impute one row,
    two-stage predict with the reject knob), timed per call, reported as
    mean/p50/p95/p99 and the implied sequential flows/s. The p95/p99 tail is
    real: it contains the flows that escalate to Stage 2.
  * batch throughput: score a whole matrix at once, at several batch sizes,
    reported as flows/s (median of several repetitions).

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

from .stream import load_bundle, load_stream

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"
BATCH_SIZES = [1024, 8192, 0]  # 0 = the whole committed sample
BATCH_REPS = 5


def _percentiles(latencies_ms: np.ndarray) -> dict:
    return {
        "mean_ms": round(float(latencies_ms.mean()), 3),
        "p50_ms": round(float(np.percentile(latencies_ms, 50)), 3),
        "p95_ms": round(float(np.percentile(latencies_ms, 95)), 3),
        "p99_ms": round(float(np.percentile(latencies_ms, 99)), 3),
        "implied_flows_per_s": round(1000.0 / float(latencies_ms.mean()), 1),
        "n_calls": int(latencies_ms.size),
    }


def bench_single_row(bundle: dict, X: np.ndarray, rows: int, seed: int) -> dict:
    """Time the serving path one flow at a time: impute the row, then the
    two-stage predict with the reject knob. This is exactly what one /predict
    request costs, so the percentiles are honest per-request numbers."""
    imputer, model = bundle["imputer"], bundle["model"]
    rng = np.random.RandomState(seed)
    idx = rng.choice(X.shape[0], size=min(rows, X.shape[0]), replace=False)

    # warmup (first joblib/sklearn call pays one-time setup costs)
    for i in idx[:3]:
        model.predict_detail(imputer.transform(X[i : i + 1]), reject_threshold=0.0)

    lat = np.empty(idx.size, dtype=float)
    for k, i in enumerate(idx):
        row = X[i : i + 1]
        t0 = time.perf_counter()
        Xi = imputer.transform(row)
        model.predict_detail(Xi, reject_threshold=0.0)
        lat[k] = (time.perf_counter() - t0) * 1000.0
    return _percentiles(lat)


def bench_batch(bundle: dict, X: np.ndarray, sizes: list[int], reps: int, seed: int) -> list[dict]:
    """Throughput of scoring a whole matrix at once (impute + two-stage predict),
    per batch size. Reports the median of `reps` runs."""
    imputer, model = bundle["imputer"], bundle["model"]
    rng = np.random.RandomState(seed)
    out = []
    for size in sizes:
        n = X.shape[0] if size <= 0 else min(size, X.shape[0])
        idx = rng.choice(X.shape[0], size=n, replace=False)
        Xb = X[idx]
        walls = []
        for _ in range(reps):
            t0 = time.perf_counter()
            Xi = imputer.transform(Xb)
            model.predict_detail(Xi, reject_threshold=0.0)
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
    bundle = load_bundle()
    X, _ = load_stream(0)
    print(f"[bench] {X.shape[0]} flows loaded; timing single-row ({rows} calls) ...")
    single = bench_single_row(bundle, X, rows, seed)
    print(
        f"[bench] single-row  mean={single['mean_ms']} ms  p50={single['p50_ms']} ms  "
        f"p95={single['p95_ms']} ms  p99={single['p99_ms']} ms  "
        f"-> {single['implied_flows_per_s']} flows/s sequential"
    )
    print("[bench] timing batch throughput ...")
    batches = bench_batch(bundle, X, BATCH_SIZES, BATCH_REPS, seed)
    for b in batches:
        print(
            f"[bench] batch {b['batch_size']:>6} rows: {b['flows_per_s']:>10,.0f} flows/s "
            f"(median of {b['reps']} reps, {b['median_wall_s']}s wall)"
        )
    report = {
        "what": (
            "measured scoring-path benchmark; single_row is the per-request serving "
            "path, batch is whole-matrix scoring"
        ),
        "environment": environment(),
        "single_row": single,
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
