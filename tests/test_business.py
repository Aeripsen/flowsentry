import json

import pytest

from flowsentry.business import (
    ARTIFACTS,
    OUT_NAME,
    REPO,
    _exact_correct,
    _exact_count,
    build_report,
    seen_traffic_table,
    wilson,
)


@pytest.fixture(scope="module")
def report():
    return build_report()


def test_committed_business_case_matches_a_fresh_derivation(report):
    """The committed file is a pure function of the committed artifacts."""
    committed = json.loads((ARTIFACTS / OUT_NAME).read_text())
    assert committed == report


def test_exact_count_recovery_refuses_when_rounding_could_lie():
    assert _exact_correct(6570, 0.8317) == 5464
    with pytest.raises(ValueError):
        _exact_correct(20_000, 0.9)


def test_seen_traffic_partitions_every_flow(report):
    n = report["full_coverage_error_split"]["n_flows"]
    for row in report["seen_traffic"]:
        assert row["answered_automatically"] + row["sent_to_analyst"] == n
        assert 0 <= row["wrong_automatic_verdicts"] <= row["answered_automatically"]


def test_exact_count_refuses_a_share_no_count_rounds_to():
    """12/100 rounds to 0.12, not 0.1235: a share that no count out of n produces
    means the inputs are inconsistent, and the recovery must not guess."""
    assert _exact_count(100, 0.12) == 12
    with pytest.raises(ValueError):
        _exact_count(100, 0.1235)


def test_per_class_analyst_counts_add_up_to_the_curve(report):
    """per_family.json (scripts/per_family_report.py) and metrics.json (train.py)
    are written by different scripts. The per-class flows sent to an analyst,
    recovered from answered shares, must sum to the curve's own count at every
    threshold both files carry."""
    curve = {r["reject_threshold"]: r for r in report["seen_traffic"]}
    checked = 0
    for row in report["analyst_share_by_class"]:
        t = row["reject_threshold"]
        if t in curve:
            assert row["all_test_flows"]["sent_to_analyst"] == curve[t]["sent_to_analyst"]
            checked += 1
    assert checked >= 3


def test_wilson_interval_known_values():
    assert wilson(0, 10) == {"ci_lower": 0.0, "ci_upper": 0.2775}
    lo_hi = wilson(32, 210)
    assert lo_hi["ci_lower"] < 32 / 210 < lo_hi["ci_upper"]
    assert wilson(5, 10) == {"ci_lower": 0.2366, "ci_upper": 0.7634}


def test_zero_day_pooled_counts_partition_the_unseen_flows(report):
    for row in report["zero_day"]["by_threshold"]:
        c = row["pooled_counts"]
        assert c["called_benign"] + c["rejected_unknown"] + c["called_wrong_attack"] == c["n"]


def test_error_split_matches_the_confusion_matrix(report):
    e = report["full_coverage_error_split"]
    assert e["benign_flows"] + e["attack_flows"] == e["n_flows"]
    # the full-coverage row of the curve counts every wrong verdict, including
    # attack-family confusions, so it bounds the two kinds split out here
    full = report["seen_traffic"][0]
    assert (
        e["false_alarms_benign_called_attack"] + e["silent_misses_attack_called_benign"]
        <= full["wrong_automatic_verdicts"]
    )


def test_table_is_built_from_counts_not_rates():
    metrics = {
        "n_test": 100,
        "coverage_reliability_curve": [
            {"threshold": 0.0, "n_covered": 100, "reliability": 0.9},
            {"threshold": 0.9, "n_covered": 40, "reliability": 1.0},
        ],
    }
    rows = seen_traffic_table(metrics)
    assert rows[0]["wrong_automatic_verdicts"] == 10
    assert rows[1]["sent_to_analyst"] == 60
    assert rows[1]["per_10k_flows"]["sent_to_analyst"] == 6000.0


def test_rarity_control_is_carried_into_every_zero_day_row(report):
    for row in report["zero_day"]["by_threshold"]:
        rc = row["rarity_control_mean_over_rounds"]
        assert 0 <= rc["seen_rare_rejected_unknown"] <= 1
        lift = rc["novelty_lift_vs_seen_rare"]
        assert lift == pytest.approx(
            row["mean_over_families"]["rejected_unknown"] - rc["seen_rare_rejected_unknown"],
            abs=2e-4,
        )


def test_readme_quotes_the_generated_resume_lines(report):
    """The resume lines are written by business.py; the README must quote them
    word for word, so a hand-edited resume claim cannot sit in the README."""
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert len(report["resume_sentences"]) == 2
    for line in report["resume_sentences"]:
        assert f"> {line}" in readme, line
