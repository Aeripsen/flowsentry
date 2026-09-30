"""Retrain the derived artifacts and require them back unchanged.
Run:  python scripts/verify_derived.py

metrics.json has scripts/reproduce.py. The business case also leans on two more
artifacts that come from their own retrains, and until this script nothing
checked that they reproduce:

  artifacts/per_family.json     scripts/per_family_report.py (one retrain)
  artifacts/zero_day_lofo.json  scripts/zero_day_lofo.py (fourteen retrains:
                                seven held-out families, real and shuffled)

Each script is rerun from the committed sample and its output compared with the
committed file as parsed JSON. The one key allowed to differ is `environment`
(a run timestamp, the interpreter and the machine); every number, every count
and the verdict text must match. The committed file is put back afterwards, so
a local run leaves the tree as it found it.

Exit 0: all match. Exit 1: something differed, with the differing top-level keys
named.
"""
from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHECKS = [
    ("scripts/per_family_report.py", "artifacts/per_family.json"),
    ("scripts/zero_day_lofo.py", "artifacts/zero_day_lofo.json"),
]
ALLOWED_TO_DIFFER = {"environment"}


def _strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in ALLOWED_TO_DIFFER}


def main() -> int:
    failed = False
    for script, artifact in CHECKS:
        path = REPO / artifact
        committed_bytes = path.read_bytes()
        committed = json.loads(committed_bytes)
        print(f"[verify] rerunning {script} ...", flush=True)
        try:
            runpy.run_path(str(REPO / script), run_name="__main__")
            regenerated = json.loads(path.read_bytes())
        finally:
            path.write_bytes(committed_bytes)
        old, new = _strip(committed), _strip(regenerated)
        if old == new:
            print(f"[verify] OK: {artifact} reproduces (ignoring {sorted(ALLOWED_TO_DIFFER)})")
            continue
        failed = True
        keys = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
        print(f"[verify] FAIL: {artifact} differs in {keys}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
