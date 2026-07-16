"""Verify the committed BCCC-UDP-QUIC sample is present.

Unlike a benchmark that must be downloaded, FlowSentry ships a stratified sample of
the public BCCC-UDP-QUIC-IDS-2025 dataset in the repo
(`data/sample/bccc_udp_quic_sample.csv.gz`, CC BY 4.0). So there is nothing to
fetch; this script just confirms the sample exists before training. To regenerate
the sample from the full dataset, see `scripts/build_sample.py` (author tool).
"""
from __future__ import annotations

import sys
from pathlib import Path

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "bccc_udp_quic_sample.csv.gz"


def main() -> int:
    if SAMPLE.exists() and SAMPLE.stat().st_size > 0:
        print(f"[ok] BCCC sample present: {SAMPLE} ({SAMPLE.stat().st_size} bytes)")
        print("     run `python -m flowsentry.train` to train + write artifacts/")
        return 0
    print(f"[error] BCCC sample missing at {SAMPLE}", file=sys.stderr)
    print("        it should ship with the repo; regenerate via scripts/build_sample.py", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
