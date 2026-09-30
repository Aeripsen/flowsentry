import json

import pytest

from flowsentry.business import (
    ARTIFACTS,
    OUT_NAME,
    _exact_correct,
    build_report,
    seen_traffic_table,
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


def test_wrong_verdicts_fall_as_the_knob_tightens(report):
    wrong = [r["wrong_automatic_verdicts"] for r in report["seen_traffic"]]
    assert wrong == sorted(wrong, reverse=True)


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


def test_nothing_is_priced(report):
    flat = json.dumps({k: v for k, v in report.items() if k != "no_prices"})
    assert "cost_" not in flat.replace("cost_on_known_traffic", "")
    assert "$" not in flat
