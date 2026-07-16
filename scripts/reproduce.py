"""Retrain from the committed sample and verify the metrics regenerate
byte-identically. Run:  python scripts/reproduce.py   (or `make reproduce`)

This is the repo's reproducibility contract: the committed
artifacts/metrics.json (binary attack PR-AUC 0.9767 and everything else) must
come back byte for byte from a fresh training run. Exact bytes are promised
under requirements.lock (the environment the numbers were published from);
under other library versions the test suite still enforces the PR-AUC floor,
but this script is the strict check.

Exit 0: identical. Exit 1: anything differed, with the differing values named.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

METRICS = Path(__file__).resolve().parents[1] / "artifacts" / "metrics.json"


def main() -> int:
    if not METRICS.exists():
        print(f"[error] {METRICS} missing; this repo commits it, check your clone")
        return 1
    committed = METRICS.read_bytes()

    from flowsentry.train import main as train_main

    print("[reproduce] retraining from the committed sample ...")
    train_main()

    regenerated = METRICS.read_bytes()
    if regenerated == committed:
        pr_auc = json.loads(regenerated)["binary_attack_detection_pr_auc"]
        print(
            f"[reproduce] OK: metrics.json is byte-identical "
            f"(binary attack PR-AUC = {pr_auc})"
        )
        return 0

    print("[reproduce] FAIL: metrics.json changed. Differing keys:")
    old, new = json.loads(committed), json.loads(regenerated)
    for key in sorted(set(old) | set(new)):
        if old.get(key) != new.get(key):
            print(f"  {key}: {old.get(key)!r} -> {new.get(key)!r}")
    print(
        "[reproduce] if you changed training on purpose, commit the new metrics.json "
        "with the change; if not, your environment differs (see requirements.lock)"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
