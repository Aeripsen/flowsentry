# ADR 004: The reject threshold is a request-time parameter

Status: accepted

## Context

A classifier that answers every flow is confidently wrong on the tail it cannot
separate (several UDP flood families look alike at the flow level). A SOC needs
the opposite failure mode: say "unknown", route to a human or a deeper detector,
and know exactly what that abstention costs in coverage.

## Decision

The model always produces (label, confidence); the reject decision compares
confidence to a threshold supplied per request (`reject_threshold` on /predict
and /predict/batch, a CLI flag on the replay, a slider on the dashboard). The
coverage-reliability curve for the whole threshold range is measured at training
time and served at /curve, so operators pick an operating point from data:
83.2% reliability at full coverage up to 99.3% at 64.8% coverage.

## Alternatives rejected

- Threshold baked in at training time: turns an operational trade-off into a
  retrain; different consumers (auto-block vs triage queue) legitimately want
  different points on the same curve at the same time.
- Calibrated probabilities as the knob: measured in
  `scripts/calibration_experiment.py`; isotonic calibration fixes the meaning of
  the number (ECE 0.041 to 0.008) but cannot re-rank flows, so the curve does
  not improve, and shipping it would cost training data. See the model card.
- Cost-sensitive learning (class weights tuned to a cost matrix): needs a cost
  matrix nobody has for a public dataset; the curve lets each deployment apply
  its own costs after the fact.

## Consequences

Every scoring surface carries the threshold parameter, and reliability numbers
must always be quoted with their coverage, never alone.
