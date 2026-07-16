"""
Batch replayer: read the committed BCCC-UDP-QUIC sample as a sequence of flows,
classify each one with the trained FlowSentry artifact, and emit a structured,
MITRE ATT&CK-tagged alert for every flow predicted as an attack (non-benign) and
not abstained.

This is a BATCH replay, not a real-time stream: it iterates a stored CSV one flow
at a time the way the deployed service sees a single request. There is no event
source, queue, or backpressure. Its purpose is honest per-flow timing: it measures
the real preprocess+classify latency (mean/p50/p95/p99) and the resulting
throughput on THIS machine. Every number printed is measured, not invented.

Run:  PYTHONPATH=src python -m flowsentry.stream --n 2000
"""
from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .attack_map import lookup
from .data import STAGE2_FEATURES, TARGET, load_sample

ARTIFACT = Path(__file__).resolve().parents[2] / "artifacts" / "flowsentry.joblib"


def load_bundle(path: Path = ARTIFACT) -> dict:
    """Load the serving artifact dict: {imputer, model, ...}."""
    if not path.exists():
        raise FileNotFoundError(
            f"model artifact missing at {path}; run `python -m flowsentry.train` first"
        )
    import joblib

    return joblib.load(path)


def load_stream(n: int):
    """Return (X_stage2, truth) for the first n sample flows (n <= 0 means all)."""
    df = load_sample()
    if n > 0:
        df = df.head(n)
    X = df[STAGE2_FEATURES].to_numpy(dtype=np.float64)
    X = np.where(np.isfinite(X), X, np.nan)
    truth = df[TARGET].to_numpy()
    return X, truth


def classify_stream(bundle: dict, X, truth, reject_threshold: float = 0.0):
    """Classify each row one flow at a time.

    Returns (alerts, latencies_ms, summary):
      alerts       list of alert dicts for attack (non-benign), non-abstained flows
      latencies_ms per-flow preprocess+classify latency in milliseconds
      summary      counts + latency percentiles for the whole run
    """
    imputer = bundle["imputer"]
    model = bundle["model"]
    n_flows = X.shape[0]
    latencies = np.empty(n_flows, dtype=float)
    alerts: list[dict] = []
    counts = {"benign": 0, "abstained": 0, "attack": 0}
    family_counts: dict[str, int] = {}

    for i in range(n_flows):
        row = X[i : i + 1]
        t0 = time.perf_counter()
        Xi = imputer.transform(row)
        labels, conf, escalated, abstained = model.predict_detail(
            Xi, reject_threshold=reject_threshold
        )
        latencies[i] = (time.perf_counter() - t0) * 1000.0

        label = str(labels[0])
        if bool(abstained[0]):
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
                "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
                "flow_index": i,
                "predicted_class": label,
                "confidence": round(float(conf[0]), 4),
                "escalated": bool(escalated[0]),
                "abstained": bool(abstained[0]),
                "mitre_id": info["technique_id"],
                "mitre_technique": info["technique_name"],
                "playbook": info["playbook"],
                "true_label": str(truth[i]),
            }
        )

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


def _format_alert(a: dict) -> str:
    mitre = f"{a['mitre_id']} {a['mitre_technique']}" if a["mitre_id"] else "n/a"
    esc = " [escalated->stage2]" if a["escalated"] else ""
    return (
        f"ALERT flow#{a['flow_index']:<5} {a['predicted_class']:<13} "
        f"conf={a['confidence']:.3f}{esc}  {mitre}  | {a['playbook']}"
    )


def run(n: int, reject_threshold: float, max_alerts: int) -> dict:
    bundle = load_bundle()
    X, truth = load_stream(n)
    print(
        f"[replay] classifying {X.shape[0]} flows from the BCCC sample "
        f"(reject_threshold={reject_threshold})\n"
    )

    wall_start = time.perf_counter()
    alerts, _latencies, summary = classify_stream(bundle, X, truth, reject_threshold)
    wall = time.perf_counter() - wall_start

    for a in alerts[:max_alerts]:
        print(_format_alert(a))
    if len(alerts) > max_alerts:
        print(f"... {len(alerts) - max_alerts} more alerts not shown")

    throughput = summary["n_flows"] / wall if wall > 0 else float("nan")
    c = summary["counts"]
    print("\n" + "=" * 70)
    print(
        f"[flows  ] {summary['n_flows']}   attacks={c['attack']}  "
        f"benign={c['benign']}  abstained={c['abstained']}"
    )
    print(f"[family ] {summary['family_counts']}")
    print(
        f"[latency] per-flow preprocess+classify  "
        f"mean={summary['mean_ms']:.3f} ms  p50={summary['p50_ms']:.3f} ms  "
        f"p95={summary['p95_ms']:.3f} ms  p99={summary['p99_ms']:.3f} ms"
    )
    print(
        f"[through] {throughput:,.0f} flows/sec over {wall:.2f}s wall "
        f"(single-thread batch replay, this machine)"
    )
    print("=" * 70)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Replay the BCCC-UDP-QUIC sample as a batch flow stream through FlowSentry."
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
    args = ap.parse_args()
    run(args.n, args.reject_threshold, args.max_alerts)


if __name__ == "__main__":
    main()
