"""
Batch replayer: read the committed BCCC-UDP-QUIC sample as a sequence of flows,
classify them with the trained FlowSentry artifact, and emit a structured,
MITRE ATT&CK-tagged alert for every flow predicted as an attack (non-benign) and
not abstained.

This is a replay of stored flows, not a live network tap. Two modes:

  per-flow (default)  scores one flow per call the way the service sees a single
                      request, and reports real per-request latency percentiles
                      (mean/p50/p95/p99) plus the implied sequential throughput.
  --batch             scores the whole matrix in one FlowScorer.score_batch call
                      and reports the measured bulk throughput.

Every number printed is measured on this machine, not invented.

Run:  python -m flowsentry.stream --n 2000
      python -m flowsentry.stream --n 0 --batch
"""
from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .attack_map import lookup
from .data import STAGE2_FEATURES, TARGET, load_sample
from .drift import drift_report
from .scoring import ARTIFACT, FlowScorer, load_bundle
from .sinks import AlertSink, JsonlSink, StdoutSink

__all__ = ["ARTIFACT", "load_bundle", "load_stream", "load_test_stream", "classify_stream"]


def load_stream(n: int):
    """Return (X_stage2, truth) for the first n sample flows (n <= 0 means all).

    This is the FULL committed sample (train + test rows). It is used only for the
    latency/throughput replay, where the mix does not matter: every printed number
    is a per-flow timing, not an accuracy claim.
    """
    df = load_sample()
    if n > 0:
        df = df.head(n)
    X = df[STAGE2_FEATURES].to_numpy(dtype=np.float64)
    X = np.where(np.isfinite(X), X, np.nan)
    truth = df[TARGET].to_numpy()
    return X, truth


def load_test_stream(bundle: dict, n: int = 0):
    """Return (X_stage2, truth) for the held-out TEST split only.

    Uses the test row positions persisted in the training artifact (`test_indices`),
    so any coverage/reliability number computed from this is measured on the exact
    same leakage-safe split `train.py` reports, NOT on the shuffled full sample.
    n <= 0 returns every test flow; n > 0 returns the first n test flows.
    """
    test_idx = bundle.get("test_indices")
    if test_idx is None:
        raise KeyError(
            "artifact has no 'test_indices'; retrain with `python -m flowsentry.train`"
        )
    df = load_sample().iloc[list(test_idx)].reset_index(drop=True)
    if n > 0:
        df = df.head(n)
    X = df[STAGE2_FEATURES].to_numpy(dtype=np.float64)
    X = np.where(np.isfinite(X), X, np.nan)
    truth = df[TARGET].to_numpy()
    return X, truth


def _build_alerts(labels, conf, escalated, abstained, truth) -> tuple[list[dict], dict, dict]:
    """Turn scored arrays into (alerts, counts, family_counts). Alerts are the
    attack (non-benign), non-abstained flows, tagged with the ATT&CK mapping."""
    alerts: list[dict] = []
    counts = {"benign": 0, "abstained": 0, "attack": 0}
    family_counts: dict[str, int] = {}
    now = datetime.now(UTC).isoformat(timespec="milliseconds")
    for i in range(len(labels)):
        label = str(labels[i])
        if bool(abstained[i]):
            counts["abstained"] += 1
            continue
        if label == "benign":
            counts["benign"] += 1
            continue
        counts["attack"] += 1
        family_counts[label] = family_counts.get(label, 0) + 1
        info = lookup(label)
        alerts.append(
            {
                "timestamp": now,
                "flow_index": i,
                "predicted_class": label,
                "confidence": round(float(conf[i]), 4),
                "escalated": bool(escalated[i]),
                "abstained": bool(abstained[i]),
                "mitre_id": info["technique_id"],
                "mitre_technique": info["technique_name"],
                "playbook": info["playbook"],
                "true_label": str(truth[i]),
            }
        )
    return alerts, counts, family_counts


def classify_stream(bundle: dict, X, truth, reject_threshold: float = 0.0):
    """Classify each row one flow at a time through the serving path.

    Returns (alerts, latencies_ms, summary):
      alerts       list of alert dicts for attack (non-benign), non-abstained flows
      latencies_ms per-flow impute+classify latency in milliseconds
      summary      counts + latency percentiles for the whole run
    """
    scorer = FlowScorer.from_bundle(bundle)
    n_flows = X.shape[0]
    latencies = np.empty(n_flows, dtype=float)
    labels = np.empty(n_flows, dtype=object)
    conf = np.empty(n_flows, dtype=float)
    escalated = np.empty(n_flows, dtype=bool)
    abstained = np.empty(n_flows, dtype=bool)

    for i in range(n_flows):
        row = X[i : i + 1]
        t0 = time.perf_counter()
        li, ci, ei, ai = scorer.score_batch(row, reject_threshold=reject_threshold)
        latencies[i] = (time.perf_counter() - t0) * 1000.0
        labels[i], conf[i], escalated[i], abstained[i] = li[0], ci[0], ei[0], ai[0]

    alerts, counts, family_counts = _build_alerts(labels, conf, escalated, abstained, truth)
    summary = {
        "n_flows": n_flows,
        "counts": counts,
        "family_counts": family_counts,
        "mean_ms": float(latencies.mean()) if n_flows else float("nan"),
        "p50_ms": float(np.percentile(latencies, 50)) if n_flows else float("nan"),
        "p95_ms": float(np.percentile(latencies, 95)) if n_flows else float("nan"),
        "p99_ms": float(np.percentile(latencies, 99)) if n_flows else float("nan"),
    }
    return alerts, latencies, summary


def classify_batch(bundle: dict, X, truth, reject_threshold: float = 0.0):
    """Classify the whole matrix in one scorer call. Returns (alerts, summary);
    the summary reports measured wall time and bulk throughput, no per-flow
    percentiles (that is what classify_stream is for)."""
    scorer = FlowScorer.from_bundle(bundle)
    t0 = time.perf_counter()
    labels, conf, escalated, abstained = scorer.score_batch(
        X, reject_threshold=reject_threshold
    )
    wall = time.perf_counter() - t0
    alerts, counts, family_counts = _build_alerts(labels, conf, escalated, abstained, truth)
    summary = {
        "n_flows": int(X.shape[0]),
        "counts": counts,
        "family_counts": family_counts,
        "wall_s": wall,
        "flows_per_s": float(X.shape[0] / wall) if wall > 0 else float("nan"),
    }
    return alerts, summary


def run(
    n: int,
    reject_threshold: float,
    max_alerts: int,
    batch: bool = False,
    jsonl: Path | None = None,
    drift: bool = False,
) -> dict:
    bundle = load_bundle()
    X, truth = load_stream(n)
    mode = "batch (one score_batch call)" if batch else "per-flow (serving path)"
    print(
        f"[replay] classifying {X.shape[0]} flows from the BCCC sample "
        f"(reject_threshold={reject_threshold}, mode={mode})\n"
    )

    if batch:
        alerts, summary = classify_batch(bundle, X, truth, reject_threshold)
    else:
        wall_start = time.perf_counter()
        alerts, _latencies, summary = classify_stream(bundle, X, truth, reject_threshold)
        summary["wall_s"] = time.perf_counter() - wall_start
        summary["flows_per_s"] = (
            summary["n_flows"] / summary["wall_s"] if summary["wall_s"] > 0 else float("nan")
        )

    sinks: list[AlertSink] = [StdoutSink(max_alerts=max_alerts)]
    if jsonl is not None:
        sinks.append(JsonlSink(jsonl))
    for a in alerts:
        for sink in sinks:
            sink.emit(a)
    for sink in sinks:
        sink.close()

    if drift:
        reference = bundle.get("drift_reference")
        if reference is None:
            print("[drift ] artifact has no drift_reference; retrain to enable --drift")
        else:
            scorer = FlowScorer.from_bundle(bundle)
            report = drift_report(reference, scorer.impute(X), scorer.feature_names)
            summary["drift"] = report
            b = report["bands"]
            print(
                f"\n[drift ] PSI vs the training distribution over this {report['n_rows']}-flow "
                f"window: {b['stable']} stable, {b['moderate']} moderate, {b['major']} major"
            )
            for row in report["top"][:5]:
                print(
                    f"[drift ]   {row['feature']:<32} psi={row['psi']:<8} {row['band']}"
                )

    c = summary["counts"]
    print("\n" + "=" * 70)
    print(
        f"[flows  ] {summary['n_flows']}   attacks={c['attack']}  "
        f"benign={c['benign']}  abstained={c['abstained']}"
    )
    print(f"[family ] {summary['family_counts']}")
    if not batch:
        print(
            f"[latency] per-flow impute+classify  "
            f"mean={summary['mean_ms']:.3f} ms  p50={summary['p50_ms']:.3f} ms  "
            f"p95={summary['p95_ms']:.3f} ms  p99={summary['p99_ms']:.3f} ms"
        )
    print(
        f"[through] {summary['flows_per_s']:,.0f} flows/sec over {summary['wall_s']:.2f}s wall "
        f"({mode}, single process, this machine)"
    )
    print("=" * 70)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Replay the BCCC-UDP-QUIC sample as a flow stream through FlowSentry."
    )
    ap.add_argument("--n", type=int, default=2000, help="flows to replay (0 = all)")
    ap.add_argument(
        "--reject-threshold",
        type=float,
        default=0.0,
        help="abstain ('unknown') when final confidence is below this",
    )
    ap.add_argument(
        "--max-alerts", type=int, default=25, help="how many alerts to print inline"
    )
    ap.add_argument(
        "--batch", action="store_true",
        help="score all flows in one vectorized call and report bulk throughput",
    )
    ap.add_argument(
        "--jsonl", type=Path, default=None,
        help="also append every alert as one JSON object per line to this file",
    )
    ap.add_argument(
        "--drift", action="store_true",
        help="report per-feature PSI of this window vs the training distribution",
    )
    args = ap.parse_args()
    run(
        args.n,
        args.reject_threshold,
        args.max_alerts,
        batch=args.batch,
        jsonl=args.jsonl,
        drift=args.drift,
    )


if __name__ == "__main__":
    main()
