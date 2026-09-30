"""The reject knob as verdicts per 10,000 flows, derived from committed artifacts.
Run:  python scripts/business_case.py [--verify]   (or `make business`)

Trains nothing. Reads artifacts/metrics.json, per_family.json and
zero_day_lofo.json and writes artifacts/business_case.json. The logic and its
limits live in src/flowsentry/business.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flowsentry.business import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
