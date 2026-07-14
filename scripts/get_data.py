"""Fetch the NSL-KDD public benchmark (KDDTrain+ / KDDTest+) into ./data.

NSL-KDD is a public research benchmark (Canadian Institute for Cybersecurity).
We use it only as a comparison baseline, never as the headline dataset.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
BASE = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master"
FILES = {"KDDTrain+.txt": f"{BASE}/KDDTrain%2B.txt", "KDDTest+.txt": f"{BASE}/KDDTest%2B.txt"}


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = DATA_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[skip] {name} already present ({dest.stat().st_size} bytes)")
            continue
        print(f"[get ] {name} <- {url}")
        urllib.request.urlretrieve(url, dest)
        print(f"[ok  ] {name} ({dest.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
