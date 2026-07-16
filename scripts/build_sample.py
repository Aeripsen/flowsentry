"""Regenerate the committed BCCC-UDP-QUIC sample from the full public dataset.

Author tool. Not needed to use the repo (the sample already ships in
`data/sample/`). It rebuilds that sample from the full
`udp_with_quic_cleaned.csv` of BCCC-UDP-QUIC-IDS-2025: keep ALL flows of the rare
UDP DDoS families and cap benign / UDP-RAW so metrics are computed on a balanced,
non-degenerate slice. Deterministic (`random_state=42`).

Only a subset of the released (CC BY 4.0) dataset is written; no non-public data.

Usage:
  python scripts/build_sample.py --source /path/to/udp_with_quic_cleaned.csv [--cap 12000]
"""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "sample" / "bccc_udp_quic_sample.csv.gz"
ID_META = ["flow_id", "timestamp", "protocol"]
LABELS = ["label", "attack_type"]
CONN = ["src_ip", "src_port", "dst_ip", "dst_port"]
FAMILIES = [
    "benign", "UDP-RAW", "UDP-VSE", "UDP-OVH", "UDP-MULTI", "UDP-HULK",
    "UDP-bypass-v1", "UDP-GAME",
]
BIG = ["UDP-RAW", "benign"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="path to udp_with_quic_cleaned.csv")
    ap.add_argument("--cap", type=int, default=12000, help="max rows for benign / UDP-RAW")
    args = ap.parse_args()

    src = Path(args.source)
    with open(src) as fh:
        header = fh.readline().strip().split(",")
    # keep the connection keys (for the split), every feature, and the target label.
    drop = set(ID_META + LABELS)
    str_cols = set(CONN + ["label_mc"])
    keep = [c for c in header if c not in drop]
    dtype = {c: (str if c in str_cols else np.float64) for c in keep}

    df = pd.read_csv(src, usecols=keep, dtype=dtype)
    df = df[df["label_mc"].isin(FAMILIES)]
    parts = [
        g.sample(min(args.cap if c in BIG else len(g), len(g)), random_state=42)
        for c, g in df.groupby("label_mc")
    ]
    sub = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT, "wt", newline="") as fh:
        sub.to_csv(fh, index=False)
    print(f"[ok] wrote {OUT} rows={len(sub)} cols={sub.shape[1]} "
          f"({OUT.stat().st_size / 1e6:.1f} MB)")
    print(sub["label_mc"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
