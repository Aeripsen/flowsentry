"""The committed Evidently page must say what the committed summary says.
Stdlib only: runs in the base test matrix without evidently installed."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_evidently_html",
                                              ROOT / "scripts" / "check_evidently_html.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
SUMMARY = json.loads(checker.SUMMARY.read_text())
PAGE = checker.extract(ROOT / SUMMARY["html_report"])


def test_committed_page_matches_committed_summary():
    assert checker.compare(SUMMARY, PAGE) == []


def test_page_carries_every_checked_column():
    assert len(PAGE["columns"]) == SUMMARY["n_features_checked"]


def test_a_summary_that_moved_is_caught():
    moved = copy.deepcopy(SUMMARY)
    col = next(iter(moved["data_drift"]["top_10_by_score"]))
    moved["data_drift"]["top_10_by_score"][col]["score"] += 0.001
    moved["performance"]["current_held_out"]["accuracy"] += 0.01
    bad = checker.compare(moved, PAGE)
    assert any(col in b for b in bad) and any("Accuracy" in b for b in bad)
