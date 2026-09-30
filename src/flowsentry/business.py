"""The reject knob in the units a SOC runs on: verdicts per 10,000 flows.

The README leads with PR-AUC, reliability and novelty lift, which are model
metrics. An analyst lead asks different questions. Of 10,000 flows, how many
does the system decide on its own, how many land in the analyst queue, and how
many of its own verdicts are wrong? When a flood family it has never seen shows
up, how many of those flows walk through as benign?

Every answer is already measured in three committed artifacts, so this module
trains nothing. It re-expresses them as counts per 10,000 flows:

  artifacts/metrics.json        coverage_reliability_curve on the held-out split
                                (pinned byte-for-byte by `make reproduce`)
  artifacts/per_family.json     the full-coverage confusion matrix and the
                                per-class answered share under the knob
  artifacts/zero_day_lofo.json  leave-one-family-out rounds, exact per-round
                                counts per threshold, and the rarity control

CI regenerates all three from the committed sample (the reproduce job) and
requires them back unchanged, so the counts here rest on retrains, not only on
arithmetic over stored JSON.

Two experiments, never mixed in one sentence:
  held-out split  one connection-grouped split, 6,570 test flows (metrics.json,
                  per_family.json). At 0.99 it answers 64.8% at 99.3% reliability.
  LOFO rounds     six retrains, each with one rare family removed
                  (zero_day_lofo.json). "Known traffic" there is each round's
                  test split minus that family, averaged over rounds: 66.5% at
                  99.4%. Similar, not the same number, and labelled apart.

Wrong verdicts are recovered exactly. The curve stores n_covered and reliability
rounded to 4 decimals, so reliability * n_covered is off from the true count of
correct answers by at most 5e-5 * n_covered (0.33 of a flow at n = 6,570).
round() gives back the exact integer while that bound is under half a flow;
`_exact_count` refuses beyond it, and also checks that the recovered count
rounds back to the stored share, which a wrong recovery could not do.

The zero-day block is checked against the artifact's own summary: the means over
the six rare families computed here from the per-round counts must equal the
summary's published means, for the unseen family and for the rarity control, or
the run fails.

What the rarity control says, and why it is here. zero_day_lofo.json's
novelty_lift compares the unseen family's rejection rate with ALL seen traffic,
which is mostly UDP-RAW and benign. The rare families that were in training
are the fair comparison, and at 0.99 they are rejected nearly as often as the
unseen one. So the knob's protection against a new flood is real (silent misses
fall), but it comes from rejecting rare floods in general, not from detecting
novelty. Every sentence built here says so.

Limits stated up front, because "per 10,000 flows" invites the reader to
picture real traffic:
  1. The held-out split is a stratified sample (benign and UDP-RAW capped, rare
     families complete), not a network's traffic mix. Per-10k rates describe
     that test mix. The knob sends 59.1% of benign flows to an analyst at 0.99
     against 35.2% of the mix, so a network with more benign traffic would send
     a LARGER share; `analyst_share_by_class` has the per-class rows.
  2. The zero-day numbers are over six rare families of 193-356 flows each, one
     round per family. Each family carries a Wilson interval (binomial sampling
     only; flows within a connection are not independent, so read it as a floor).

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
Z95 = 1.959963984540054


def _per(k: int, n: int) -> float:
    return round(k * PER / n, 1) if n else 0.0


def _exact_count(n: int, share: float) -> int:
    """The integer count behind a share stored to 4 decimals. Exact while the
    rounding error, at most 5e-5 * n, stays under half a unit; the round trip
    back to the stored share is checked as well."""
    if n * 5e-5 >= 0.5:
        raise ValueError(
            f"n={n} is too large to recover exact counts from a 4-decimal share"
        )
    k = round(share * n)
    if n and round(k / n, 4) != round(share, 4):
        raise ValueError(f"share {share} is not a 4-decimal rounding of any count out of {n}")
    return k


# kept under its old name: the curve's recovery of correct answers
_exact_correct = _exact_count


def wilson(k: int, n: int) -> dict[str, float]:
    """Wilson 95% interval for k successes in n trials."""
    if n == 0:
        return {"ci_lower": 0.0, "ci_upper": 1.0}
    ph = k / n
    den = 1 + Z95**2 / n
    centre = (ph + Z95**2 / (2 * n)) / den
    half = Z95 * math.sqrt(ph * (1 - ph) / n + Z95**2 / (4 * n * n)) / den
    return {
        "ci_lower": round(max(0.0, centre - half), 4),
        "ci_upper": round(min(1.0, centre + half), 4),
    }


def seen_traffic_table(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """The coverage-reliability curve as verdicts per 10,000 held-out flows."""
    n = int(metrics["n_test"])
    rows = []
    for r in metrics["coverage_reliability_curve"]:
        covered = int(r["n_covered"])
        wrong = covered - _exact_count(covered, float(r["reliability"]))
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


def analyst_share_by_class(per_family: dict[str, Any]) -> list[dict[str, Any]]:
    """Per class, how many held-out flows the knob sends to an analyst. The
    per-10k rates above describe the stratified mix; these rows are what a
    reader needs to re-weight them to any other mix."""
    out = []
    for row in per_family["per_family_under_reject"]:
        classes = {}
        for cls, v in row["per_family"].items():
            n = int(v["support"])
            answered = _exact_count(n, float(v["answered_share"]))
            classes[cls] = {
                "n_flows": n,
                "answered": answered,
                "sent_to_analyst": n - answered,
                "sent_to_analyst_share": round((n - answered) / n, 4),
            }
        n_all = sum(c["n_flows"] for c in classes.values())
        to_analyst = sum(c["sent_to_analyst"] for c in classes.values())
        out.append(
            {
                "reject_threshold": row["reject_threshold"],
                "all_test_flows": {
                    "n_flows": n_all,
                    "sent_to_analyst": to_analyst,
                    "sent_to_analyst_share": round(to_analyst / n_all, 4),
                },
                "per_class": classes,
            }
        )
    return out


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


def _mean(values: list[float]) -> float:
    return round(math.fsum(values) / len(values), 4)


def zero_day_table(lofo: dict[str, Any], arm: str = ZERO_DAY_ARM) -> dict[str, Any]:
    """Per threshold, what happens to flows of a family the model never saw:
    pooled counts over the rare-family rounds, the mean over families that the
    artifact's summary publishes (and that this function re-derives), a Wilson
    interval per family, and the same numbers for the rare families that WERE
    in training in the same rounds (the rarity control)."""
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
        ctrl: dict[str, list[float]] = {"called_benign": [], "rejected_unknown": []}
        lift_vs_rare: list[float] = []
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
            sr = row["seen_rare_families"]
            for k in ctrl:
                ctrl[k].append(float(sr[k]))
            lift_vs_rare.append(float(row["novelty_lift_vs_seen_rare"]))
            seen_cov.append(float(row["seen_families"]["coverage"]))
            seen_rel.append(float(row["seen_families"]["reliability"]))
            per_family[r["held_out_family"]] = {
                "n_unseen_flows": int(u["n"]),
                "called_benign": u["called_benign"],
                "called_benign_ci95": wilson(int(u["n_called_benign"]), int(u["n"])),
                "rejected_unknown": u["rejected_unknown"],
                "rejected_unknown_ci95": wilson(int(u["n_rejected"]), int(u["n"])),
                "seen_rare_rejected_unknown": sr["rejected_unknown"],
                "seen_rare_called_benign": sr["called_benign"],
            }
        n = pooled["n"]
        table.append(
            {
                "reject_threshold": t,
                "mean_over_families": {k: _mean(v) for k, v in shares.items()},
                "pooled_counts": pooled,
                "pooled_ci95": {
                    "called_benign": wilson(pooled["called_benign"], n),
                    "rejected_unknown": wilson(pooled["rejected_unknown"], n),
                },
                "pooled_per_10k_unseen_flows": {
                    "passed_as_benign": _per(pooled["called_benign"], n),
                    "sent_to_analyst_as_unknown": _per(pooled["rejected_unknown"], n),
                    "labelled_as_a_different_attack": _per(pooled["called_wrong_attack"], n),
                },
                "rarity_control_mean_over_rounds": {
                    "seen_rare_called_benign": _mean(ctrl["called_benign"]),
                    "seen_rare_rejected_unknown": _mean(ctrl["rejected_unknown"]),
                    "novelty_lift_vs_seen_rare": _mean(lift_vs_rare),
                },
                "cost_on_known_traffic": {
                    "mean_seen_coverage": _mean(seen_cov),
                    "mean_seen_reliability": _mean(seen_rel),
                },
                "per_family": per_family,
            }
        )
    return {"arm": arm, "rare_families": rare, "by_threshold": table}


def check_zero_day_against_summary(zd: dict[str, Any], lofo: dict[str, Any]) -> str:
    """The means derived here must equal the ones zero_day_lofo.py published."""
    s = lofo["summary"]
    arm = zd["arm"]
    forced = s["forced_to_answer_threshold_0"]["per_arm"][arm]["mean_called_benign"]
    head = s[f"at_threshold_{HEADLINE_THRESHOLD}"]["per_arm"][arm]
    rc = s["rarity_control"]["per_arm"][arm]
    rc0, rch = rc["threshold_0"], rc[f"threshold_{HEADLINE_THRESHOLD}"]
    by_t = {row["reject_threshold"]: row for row in zd["by_threshold"]}
    z0, zh = by_t[0.0], by_t[HEADLINE_THRESHOLD]
    pairs = [
        (z0["mean_over_families"]["called_benign"], forced, "forced called_benign"),
        (zh["mean_over_families"]["called_benign"], head["mean_called_benign"],
         "called_benign at headline"),
        (zh["mean_over_families"]["rejected_unknown"], head["mean_rejected_unknown"],
         "rejected_unknown at headline"),
        (zh["cost_on_known_traffic"]["mean_seen_coverage"], head["mean_seen_coverage"],
         "seen coverage at headline"),
        (zh["cost_on_known_traffic"]["mean_seen_reliability"], head["mean_seen_reliability"],
         "seen reliability at headline"),
        (z0["rarity_control_mean_over_rounds"]["seen_rare_called_benign"],
         rc0["mean_seen_rare_called_benign"], "forced seen-rare called_benign"),
        (zh["rarity_control_mean_over_rounds"]["seen_rare_called_benign"],
         rch["mean_seen_rare_called_benign"], "seen-rare called_benign at headline"),
        (zh["rarity_control_mean_over_rounds"]["seen_rare_rejected_unknown"],
         rch["mean_seen_rare_rejected_unknown"], "seen-rare rejected at headline"),
        (zh["rarity_control_mean_over_rounds"]["novelty_lift_vs_seen_rare"],
         rch["mean_novelty_lift_vs_seen_rare"], "rarity-controlled lift at headline"),
    ]
    # Both sides are 4-decimal roundings of the same mean, summed by np.mean there
    # and math.fsum here, so they may differ by one unit in the last place when the
    # mean sits on a rounding boundary. Anything larger is a real disagreement.
    for mine, theirs, what in pairs:
        if round(abs(mine - theirs), 6) > 1e-4:
            raise SystemExit(f"FAIL: {what}: derived {mine}, summary says {theirs}")
    return f"zero-day means match zero_day_lofo.json summary on {len(pairs)} published values"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _parts(report: dict[str, Any]) -> dict[str, Any]:
    seen = {r["reject_threshold"]: r for r in report["seen_traffic"]}
    zd = {r["reject_threshold"]: r for r in report["zero_day"]["by_threshold"]}
    by_class = {r["reject_threshold"]: r for r in report["analyst_share_by_class"]}
    zh = zd[HEADLINE_THRESHOLD]
    worst = max(zh["per_family"].items(), key=lambda kv: kv[1]["called_benign"])
    return {
        "full": seen[0.0], "head": seen[HEADLINE_THRESHOLD],
        "z0": zd[0.0], "zh": zh,
        "cls": by_class[HEADLINE_THRESHOLD],
        "n_fam": len(report["zero_day"]["rare_families"]),
        "worst": worst,
        "n_test": report["full_coverage_error_split"]["n_flows"],
    }


def headline_sentences(report: dict[str, Any]) -> list[str]:
    """The README sentences. Each one uses a single experiment and says which."""
    p = _parts(report)
    full, head, z0, zh, cls = p["full"], p["head"], p["z0"], p["zh"], p["cls"]
    rc0, rch = z0["rarity_control_mean_over_rounds"], zh["rarity_control_mean_over_rounds"]
    wname, w = p["worst"]
    benign = cls["per_class"]["benign"]
    return [
        (
            f"Leave-one-family-out ({p['n_fam']} rare UDP DDoS families, one retrain each): "
            f"forced to answer, the model passes {_pct(z0['mean_over_families']['called_benign'])} "
            f"of a never-seen family's flows as benign. With the reject option at "
            f"{HEADLINE_THRESHOLD} that falls to "
            f"{_pct(zh['mean_over_families']['called_benign'])}, "
            f"because {_pct(zh['mean_over_families']['rejected_unknown'])} go to an analyst as "
            f"unknown. The worst family, {wname}, still passes "
            f"{_pct(w['called_benign'])} as benign (95% CI "
            f"{_pct(w['called_benign_ci95']['ci_lower'])} to "
            f"{_pct(w['called_benign_ci95']['ci_upper'])})."
        ),
        (
            f"That is not novelty detection. In the same rounds, the rare families that WERE in "
            f"training are passed as benign {_pct(rc0['seen_rare_called_benign'])} of the time "
            f"with no knob and {_pct(rch['seen_rare_called_benign'])} at {HEADLINE_THRESHOLD}, "
            f"where {_pct(rch['seen_rare_rejected_unknown'])} of them go to an analyst too: the "
            f"lift over them is {rch['novelty_lift_vs_seen_rare']:.3f}. The knob protects "
            f"against a new flood by rejecting rare floods in general."
        ),
        (
            f"Held-out split ({p['n_test']:,} flows, stratified test mix): wrong automatic "
            f"verdicts fall from {full['per_10k_flows']['wrong_automatic_verdicts']:,.0f} to "
            f"{head['per_10k_flows']['wrong_automatic_verdicts']:,.0f} per 10,000 flows "
            f"({full['wrong_automatic_verdicts']:,} to {head['wrong_automatic_verdicts']:,}) at "
            f"{HEADLINE_THRESHOLD}, with {_pct(head['sent_to_analyst'] / p['n_test'])} of flows "
            f"sent to an analyst. That share is {_pct(benign['sent_to_analyst_share'])} for "
            f"benign flows, so a network with more benign traffic than this capped sample "
            f"would send more, not less."
        ),
    ]


def resume_sentences(report: dict[str, Any]) -> list[str]:
    """The exact resume lines, one experiment per line. tests/test_business.py
    holds the README to them word for word."""
    p = _parts(report)
    full, head, z0, zh, cls = p["full"], p["head"], p["z0"], p["zh"], p["cls"]
    rch = zh["rarity_control_mean_over_rounds"]
    benign = cls["per_class"]["benign"]
    return [
        (
            f"Measured open-set behaviour with leave-one-family-out retrains over "
            f"{p['n_fam']} rare UDP DDoS families plus a rarity control: forced to answer, the "
            f"model passes {_pct(z0['mean_over_families']['called_benign'])} of a never-seen "
            f"family as benign; a {HEADLINE_THRESHOLD} reject option cuts that to "
            f"{_pct(zh['mean_over_families']['called_benign'])} by routing "
            f"{_pct(zh['mean_over_families']['rejected_unknown'])} to an analyst, and the "
            f"control shows trained-on rare families routed almost as often "
            f"({_pct(rch['seen_rare_rejected_unknown'])}), so the gain comes from rejecting "
            f"rare floods, not from detecting novelty."
        ),
        (
            f"On a stratified held-out split of {p['n_test']:,} flows, a {HEADLINE_THRESHOLD} "
            f"reject threshold cut wrong automatic verdicts from "
            f"{full['per_10k_flows']['wrong_automatic_verdicts']:,.0f} to "
            f"{head['per_10k_flows']['wrong_automatic_verdicts']:,.0f} per 10,000 flows by "
            f"sending {_pct(head['sent_to_analyst'] / p['n_test'])} of flows to an analyst, "
            f"{_pct(benign['sent_to_analyst_share'])} of benign ones."
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
            "analyst, wrong; and what happens to flows of a never-seen attack family, "
            "next to rare families that were in training"
        ),
        "derived_from": [
            "artifacts/metrics.json (make reproduce)",
            "artifacts/per_family.json (make families)",
            "artifacts/zero_day_lofo.json (make zero-day)",
        ],
        "no_prices": "counts only; no analyst-time or breach-cost figure is used",
        "test_mix_caveat": (
            "the held-out split is a stratified sample with benign and UDP-RAW "
            "capped; per-10k rates describe that mix, not a real network's. At the "
            "headline threshold the knob sends a larger share of benign flows to an "
            "analyst than of the mix, so a benign-heavier network sends more; see "
            "analyst_share_by_class"
        ),
        "experiments": {
            "held_out_split": "seen_traffic, analyst_share_by_class, full_coverage_error_split",
            "leave_one_family_out": "zero_day (unseen family, rarity control, known-traffic "
                                    "cost as the mean over that experiment's rounds)",
        },
        "consistency_check": check,
        "headline_threshold": HEADLINE_THRESHOLD,
        "seen_traffic": seen_traffic_table(metrics),
        "analyst_share_by_class": analyst_share_by_class(per_family),
        "full_coverage_error_split": full_coverage_errors(per_family),
        "zero_day": zd,
    }
    report["headline_sentences"] = headline_sentences(report)
    report["resume_sentences"] = resume_sentences(report)
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
    for s in report["resume_sentences"]:
        print(f"[cv   ] {s}")
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
