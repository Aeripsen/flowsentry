"""Export the held-out flows the live demo page runs on: `python scripts/demo_data.py`.

The pipeline lives in src/flowsentry/demo.py; it refuses to write unless the
exported rows rebuild the committed curve, binary PR-AUC and per-family confusion.
`--verify` rebuilds the export from a retrain and compares it to the committed
file byte for byte instead of writing.
"""
from __future__ import annotations

import sys

from flowsentry.demo import main

if __name__ == "__main__":
    main(sys.argv[1:])
