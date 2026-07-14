"""
Streaming replayer: treat KDDTest+ as a live flow stream, classify each flow with
the trained FlowSentry artifact, and emit a structured alert for every flow that is
predicted as an attack (non-normal) and not abstained.

The point of this file is honest per-flow timing. It classifies one flow at a time,
the way the deployed service sees traffic, and measures the real preprocess+classify
latency (mean/p50/p95/p99) and throughput. Numbers printed at the end are measured on
THIS machine over the flows actually replayed. Nothing here is invented.

Run:  PYTHONPATH=src python -m flowsentry.stream --n 2000
"""
from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .attack_map import lookup
from .data import ATTACK_CATEGORY, COLUMNS

ARTIFACT = Path(__file__).resolve().parents[2] / "artifacts" / "flowsentry.joblib"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_bundle(path: Path = ARTIFACT) -> dict:
    """Load the serving artifact dict: {preprocessor, model, feature_names}."""
    if not path.exists():
        raise FileNotFoundError(
            f"model artifact missing at {path}; run `python -m flowsentry.train` first"
        )
    return joblib.load(path)


def load_stream(n: int, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Read the first n rows of KDDTest+ as raw flow records (n <= 0 means all)."""
    path = data_dir / "KDDTest+.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"test data missing at {path}; run `python scripts/get_data.py` first"
        )
    df = pd.read_csv(path, header=None, names=COLUMNS, nrows=n if n > 0 else None)
    # ground-truth 5-class category, kept only so the demo can show truth beside the guess
    df["category"] = df["label"].map(ATTACK_CATEGORY).fillna("unknown_attack")
    return df


def classify_stream(bundle: dict, df: pd.DataFrame, reject_threshold: float = 0.0):
    """Classify each row one flow at a time.

    Returns (alerts, latencies_ms, summary):
      alerts       list of alert dicts for non-normal, non-abstained flows
      latencies_ms per-flow preprocess+classify latency in milliseconds (np.ndarray)
      summary      counts + latency percentiles for the whole run
    """
    pre = bundle["preprocessor"]
    model = bundle["model"]
    n_flows = len(df)
    latencies = np.empty(n_flows, dtype=float)
    alerts: list[dict] = []
    counts = {"normal": 0, "abstained": 0, "attack": 0}
    family_counts: dict[str, int] = {}
    has_truth = "category" in df.columns

    for i in range(n_flows):
        row = df.iloc[[i]]  # 1-row frame keeps column names/dtypes for the preprocessor
        t0 = time.perf_counter()
        X = pre.transform(row)
        labels, conf, escalated, abstained = model.predict_detail(
            X, reject_threshold=reject_threshold
        )
        latencies[i] = (time.perf_counter() - t0) * 1000.0

        label = str(labels[0])
        if bool(abstained[0]):
            counts["abstained"] += 1
            continue
        if label == "normal":
            counts["normal"] += 1
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
                "true_category": str(row["category"].iloc[0]) if has_truth else None,
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
        f"ALERT flow#{a['flow_index']:<5} {a['predicted_class']:<6} "
        f"conf={a['confidence']:.3f}{esc}  {mitre}  | {a['playbook']}"
    )


def run(n: int, reject_threshold: float, max_alerts: int) -> dict:
    bundle = load_bundle()
    df = load_stream(n)
    print(
        f"[stream] replaying {len(df)} flows from KDDTest+ "
        f"(reject_threshold={reject_threshold})\n"
    )

    wall_start = time.perf_counter()
    alerts, _latencies, summary = classify_stream(bundle, df, reject_threshold)
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
        f"normal={c['normal']}  abstained={c['abstained']}"
    )
    print(f"[family ] {summary['family_counts']}")
    print(
        f"[latency] per-flow preprocess+classify  "
        f"mean={summary['mean_ms']:.3f} ms  p50={summary['p50_ms']:.3f} ms  "
        f"p95={summary['p95_ms']:.3f} ms  p99={summary['p99_ms']:.3f} ms"
    )
    print(
        f"[through] {throughput:,.0f} flows/sec over {wall:.2f}s wall "
        f"(single-thread, this machine)"
    )
    print("=" * 70)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Replay KDDTest+ as a live flow stream through FlowSentry."
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
