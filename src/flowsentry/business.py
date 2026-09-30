"""The reject knob in the units a SOC runs on: verdicts per 10,000 flows.

The README leads with PR-AUC, reliability and novelty lift, which are model
metrics. An analyst lead asks different questions. Of 10,000 flows, how many
does the system decide on its own, how many land in the analyst queue, and how
many of its own verdicts are wrong? When a flood family it has never seen shows
up, how many of those flows walk through as benign?

Every answer is already measured in two committed artifacts, so this module
trains nothing. It re-expresses them as counts per 10,000 flows:

  artifacts/metrics.json        coverage_reliability_curve on the held-out split
                                (pinned byte-for-byte by `make reproduce`)
  artifacts/per_family.json     the full-coverage confusion matrix
  artifacts/zero_day_lofo.json  leave-one-family-out rounds, exact per-round
                                counts per threshold (`make zero-day`)

Wrong verdicts are recovered exactly. The curve stores n_covered and reliability
rounded to 4 decimals; n_covered <= 6,570, so the rounding moves
reliability * n_covered by at most 0.33 of a flow, and round() gives back the
exact integer count of correct answers. `_exact_correct` asserts that bound.

The zero-day block is checked against the artifact's own summary: the mean over
the six rare families computed here from the per-round counts must equal the
summary's published means, or the run fails.

Two limits stated up front, because "per 10,000 flows" invites the reader to
picture real traffic:
  1. The held-out split is a stratified sample (benign and UDP-RAW capped, rare
     families complete), not a network's traffic mix. Per-10k rates describe
     that test mix. A real network with more benign traffic would see different
     queue sizes.
  2. The zero-day means are over six rare families of 207-383 flows each, one
     round per family. They are measurements on this dataset, not a guarantee
     about the next attack family.

Means over families use math.fsum, not sum(): Python 3.12 made sum() of floats
compensated, so on 3.11 a plain sum lands a few means on the other side of a
4-decimal rounding boundary. fsum is correctly rounded on every version, which
keeps the committed file byte-identical across the CI matrix.

Nothing is priced. There is no analyst-minute or breach-cost figure anywhere.

Run:  python scripts/business_case.py            -> artifacts/business_case.json
      python scripts/business_case.py --verify   require a byte match
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO / "artifacts"
OUT_NAME = "business_case.json"

PER = 10_000
ZERO_DAY_ARM = "hierarchy"  # the architecture the service ships
HEADLINE_THRESHOLD = 0.99   # the README's 99.3% reliability operating point


def _per(k: int, n: int) -> float:
    return round(k * PER / n, 1) if n else 0.0


def _exact_correct(n_covered: int, reliability: float) -> int:
    """Correct answers from a 4-decimal reliability. Exact while the rounding
    error, at most 5e-5 * n_covered, stays under half a flow."""
    if n_covered * 5e-5 >= 0.5:
        raise ValueError(
            f"n_covered={n_covered} is too large to recover exact counts from a "
            "4-decimal reliability"
        )
    return round(reliability * n_covered)


def seen_traffic_table(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """The coverage-reliability curve as verdicts per 10,000 held-out flows."""
    n = int(metrics["n_test"])
    rows = []
    for r in metrics["coverage_reliability_curve"]:
        covered = int(r["n_covered"])
        wrong = covered - _exact_correct(covered, float(r["reliability"]))
        to_analyst = n - covered
        rows.append(
            {
                "reject_threshold": r["threshold"],
                "answered_automatically": covered,
                "sent_to_analyst": to_analyst,
                "wrong_automatic_verdicts": wrong,
                "per_10k_flows": {
                    "answered_automatically": _per(covered, n),
                    "sent_to_analyst": _per(to_analyst, n),
                    "wrong_automatic_verdicts": _per(wrong, n),
                },
                "reliability_of_answered": r["reliability"],
            }
        )
    return rows


def full_coverage_errors(per_family: dict[str, Any]) -> dict[str, Any]:
    """With no reject knob, split the wrong verdicts into the two a SOC feels
    differently: benign flows raised as attacks (false alarms) and attack flows
    passed as benign (silent misses)."""
    conf = per_family["confusion_full_coverage"]
    n = int(per_family["n_test"])
    benign = conf["benign"]
    false_alarms = int(benign["n_flows"]) - int(benign["called"].get("benign", 0))
    attack_flows = sum(int(v["n_flows"]) for k, v in conf.items() if k != "benign")
    silent = sum(int(v["called"].get("benign", 0)) for k, v in conf.items() if k != "benign")
    return {
        "n_flows": n,
        "benign_flows": int(benign["n_flows"]),
        "false_alarms_benign_called_attack": false_alarms,
        "false_alarm_share_of_benign": round(false_alarms / int(benign["n_flows"]), 4),
        "attack_flows": attack_flows,
        "silent_misses_attack_called_benign": silent,
        "silent_miss_share_of_attacks": round(silent / attack_flows, 4),
        "per_10k_flows": {
            "false_alarms": _per(false_alarms, n),
            "silent_misses": _per(silent, n),
        },
        "note": "full coverage only; the committed artifacts do not split wrong "
                "verdicts by kind under the reject knob, so this table does not either",
    }


def zero_day_table(lofo: dict[str, Any], arm: str = ZERO_DAY_ARM) -> dict[str, Any]:
    """Per threshold, what happens to flows of a family the model never saw:
    pooled counts over the rare-family rounds, plus the mean over families that
    the artifact's summary publishes (and that this function re-derives)."""
    rare = list(lofo["summary"]["rare_families"])
    rounds = [r for r in lofo["rounds"] if r["held_out_family"] in rare]
    if sorted(r["held_out_family"] for r in rounds) != sorted(rare):
        raise ValueError("zero_day_lofo.json is missing a rare-family round")
    thresholds = [row["threshold"] for row in rounds[0]["arms"][arm]["by_threshold"]]
    table = []
    for t in thresholds:
        pooled = {"n": 0, "called_benign": 0, "rejected_unknown": 0, "called_wrong_attack": 0}
        shares: dict[str, list[float]] = {
            "called_benign": [], "rejected_unknown": [], "called_wrong_attack": []
        }
        seen_cov: list[float] = []
        seen_rel: list[float] = []
        per_family = {}
        for r in rounds:
            row = next(x for x in r["arms"][arm]["by_threshold"] if x["threshold"] == t)
            u = row["unseen_family"]
            pooled["n"] += int(u["n"])
            pooled["called_benign"] += int(u["n_called_benign"])
            pooled["rejected_unknown"] += int(u["n_rejected"])
            pooled["called_wrong_attack"] += int(u["n_called_wrong_attack"])
            for k in shares:
                shares[k].append(float(u[k]))
            seen_cov.append(float(row["seen_families"]["coverage"]))
            seen_rel.append(float(row["seen_families"]["reliability"]))
            per_family[r["held_out_family"]] = {
                "n_unseen_flows": int(u["n"]),
                "called_benign": u["called_benign"],
                "rejected_unknown": u["rejected_unknown"],
            }
        n = pooled["n"]
        table.append(
            {
                "reject_threshold": t,
                "mean_over_families": {
                    k: round(math.fsum(v) / len(v), 4) for k, v in shares.items()
                },
                "pooled_counts": pooled,
                "pooled_per_10k_unseen_flows": {
                    "passed_as_benign": _per(pooled["called_benign"], n),
                    "sent_to_analyst_as_unknown": _per(pooled["rejected_unknown"], n),
                    "labelled_as_a_different_attack": _per(pooled["called_wrong_attack"], n),
                },
                "cost_on_known_traffic": {
                    "mean_seen_coverage": round(math.fsum(seen_cov) / len(seen_cov), 4),
                    "mean_seen_reliability": round(math.fsum(seen_rel) / len(seen_rel), 4),
                },
                "per_family": per_family,
            }
        )
    return {"arm": arm, "rare_families": rare, "by_threshold": table}


def check_zero_day_against_summary(zd: dict[str, Any], lofo: dict[str, Any]) -> str:
    """The means derived here must equal the ones zero_day_lofo.py published."""
    s = lofo["summary"]
    forced = s["forced_to_answer_threshold_0"]["per_arm"][zd["arm"]]["mean_called_benign"]
    head = s[f"at_threshold_{HEADLINE_THRESHOLD}"]["per_arm"][zd["arm"]]
    by_t = {row["reject_threshold"]: row for row in zd["by_threshold"]}
    pairs = [
        (by_t[0.0]["mean_over_families"]["called_benign"], forced, "forced called_benign"),
        (by_t[HEADLINE_THRESHOLD]["mean_over_families"]["called_benign"],
         head["mean_called_benign"], "called_benign at headline"),
        (by_t[HEADLINE_THRESHOLD]["mean_over_families"]["rejected_unknown"],
         head["mean_rejected_unknown"], "rejected_unknown at headline"),
        (by_t[HEADLINE_THRESHOLD]["cost_on_known_traffic"]["mean_seen_coverage"],
         head["mean_seen_coverage"], "seen coverage at headline"),
        (by_t[HEADLINE_THRESHOLD]["cost_on_known_traffic"]["mean_seen_reliability"],
         head["mean_seen_reliability"], "seen reliability at headline"),
    ]
    for mine, theirs, what in pairs:
        if abs(mine - theirs) > 1e-4:
            raise SystemExit(f"FAIL: {what}: derived {mine}, summary says {theirs}")
    return f"zero-day means match zero_day_lofo.json summary on {len(pairs)} published values"


def headline_sentences(report: dict[str, Any]) -> list[str]:
    seen = {r["reject_threshold"]: r for r in report["seen_traffic"]}
    full, head = seen[0.0], seen[HEADLINE_THRESHOLD]
    zd = {r["reject_threshold"]: r for r in report["zero_day"]["by_threshold"]}
    z0, zh = zd[0.0], zd[HEADLINE_THRESHOLD]

    def pct(x: float) -> str:
        return f"{100 * x:.1f}%"

    return [
        (
            f"On attack families held out of training entirely (leave-one-family-out, "
            f"{len(report['zero_day']['rare_families'])} families), the reject option at "
            f"{HEADLINE_THRESHOLD} cuts silent misses (attack flows passed as benign) from "
            f"{pct(z0['mean_over_families']['called_benign'])} to "
            f"{pct(zh['mean_over_families']['called_benign'])} and sends "
            f"{pct(zh['mean_over_families']['rejected_unknown'])} to an analyst as unknown, "
            f"while answering {pct(zh['cost_on_known_traffic']['mean_seen_coverage'])} of "
            f"known traffic at {pct(zh['cost_on_known_traffic']['mean_seen_reliability'])} "
            f"reliability."
        ),
        (
            f"On the held-out split, wrong automatic verdicts fall from "
            f"{full['per_10k_flows']['wrong_automatic_verdicts']:,.0f} to "
            f"{head['per_10k_flows']['wrong_automatic_verdicts']:,.0f} per 10,000 flows "
            f"({full['wrong_automatic_verdicts']:,} to {head['wrong_automatic_verdicts']:,} of "
            f"{full['answered_automatically']:,}), with "
            f"{head['per_10k_flows']['sent_to_analyst']:,.0f} per 10,000 sent to an analyst."
        ),
    ]


def build_report(artifacts: Path = ARTIFACTS) -> dict[str, Any]:
    metrics = json.loads((artifacts / "metrics.json").read_text())
    per_family = json.loads((artifacts / "per_family.json").read_text())
    lofo = json.loads((artifacts / "zero_day_lofo.json").read_text())
    if int(per_family["n_test"]) != int(metrics["n_test"]):
        raise SystemExit("FAIL: per_family.json and metrics.json describe different splits")

    zd = zero_day_table(lofo)
    check = check_zero_day_against_summary(zd, lofo)
    report: dict[str, Any] = {
        "what": (
            "the reject knob as verdicts per 10,000 flows: answered, sent to an "
            "analyst, wrong; and what happens to flows of a never-seen attack family"
        ),
        "derived_from": [
            "artifacts/metrics.json (make reproduce)",
            "artifacts/per_family.json (make families)",
            "artifacts/zero_day_lofo.json (make zero-day)",
        ],
        "no_prices": "counts only; no analyst-time or breach-cost figure is used",
        "test_mix_caveat": (
            "the held-out split is a stratified sample with benign and UDP-RAW "
            "capped; per-10k rates describe that mix, not a real network's"
        ),
        "consistency_check": check,
        "headline_threshold": HEADLINE_THRESHOLD,
        "seen_traffic": seen_traffic_table(metrics),
        "full_coverage_error_split": full_coverage_errors(per_family),
        "zero_day": zd,
    }
    report["headline_sentences"] = headline_sentences(report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FlowSentry business-outcome table.")
    ap.add_argument("--verify", action="store_true",
                    help="regenerate and fail unless byte-identical to the committed file")
    args = ap.parse_args(argv)

    report = build_report()
    payload = json.dumps(report, indent=2) + "\n"
    out = ARTIFACTS / OUT_NAME
    print(f"[check] {report['consistency_check']}")
    for s in report["headline_sentences"]:
        print(f"[line ] {s}")
    if args.verify:
        if not out.exists():
            print(f"FAIL: {out} missing")
            return 1
        if out.read_bytes().replace(b"\r\n", b"\n") != payload.encode():
            print(f"FAIL: regenerated {OUT_NAME} differs from the committed file")
            return 1
        print(f"PASS: {OUT_NAME} regenerates byte-identically from the committed artifacts")
        return 0
    out.write_text(payload, newline="\n")
    print(f"[save ] {out}")
    return 0
