"""The live demo page cannot drift from the committed artifacts.

The committed per-flow export must rebuild the committed coverage-reliability
curve and binary PR-AUC (metrics.json) and the full-coverage confusion counts
(per_family.json). The zero-day panel computes plain means over the six rare
families from zero_day_lofo.json's rounds, so those means must equal the
committed summary. pages.yml runs this file before every deploy."""
import json
import statistics
from pathlib import Path

import pytest

from flowsentry.demo import check_export

ART = Path(__file__).resolve().parents[1] / "artifacts"


def _load(name):
    path = ART / name
    if not path.exists():
        pytest.skip(f"{name} not committed")
    return json.loads(path.read_text())


def test_demo_export_rebuilds_committed_numbers():
    errors = check_export(
        _load("demo_flows.json"), _load("metrics.json"), _load("per_family.json")
    )
    assert errors == []


@pytest.mark.parametrize(
    ("threshold", "summary_key", "field", "stat"),
    [
        (0.0, "forced_to_answer_threshold_0", "called_benign", "mean_called_benign"),
        (0.99, "at_threshold_0.99", "called_benign", "mean_called_benign"),
        (0.99, "at_threshold_0.99", "rejected_unknown", "mean_rejected_unknown"),
    ],
)
def test_zero_day_means_on_the_page_match_the_summary(threshold, summary_key, field, stat):
    z = _load("zero_day_lofo.json")
    rare = z["summary"]["rare_families"]
    rates = [
        next(b for b in r["arms"]["hierarchy"]["by_threshold"] if b["threshold"] == threshold)[
            "unseen_family"
        ][field]
        for r in z["rounds"]
        if r["held_out_family"] in rare
    ]
    expected = z["summary"][summary_key]["per_arm"]["hierarchy"][stat]
    assert round(statistics.mean(rates), 4) == expected


def test_readme_top_block_quotes_the_artifacts():
    z = _load("zero_day_lofo.json")["summary"]
    m = _load("metrics.json")
    readme = (ART.parent / "README.md").read_text(encoding="utf-8")
    forced = z["forced_to_answer_threshold_0"]["per_arm"]["hierarchy"]["mean_called_benign"]
    knob = z["at_threshold_0.99"]["per_arm"]["hierarchy"]
    row = next(r for r in m["coverage_reliability_curve"] if r["threshold"] == 0.99)
    expected = [
        f"from **{forced:.1%} to {knob['mean_called_benign']:.1%}**",
        f"**{knob['mean_rejected_unknown']:.1%}** are sent to an analyst as unknown",
        f"answers **{row['coverage']:.1%}** of held-out traffic at **{row['reliability']:.1%}**",
    ]
    for line in expected:
        assert line in readme, line
