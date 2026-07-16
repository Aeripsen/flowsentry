# ADR 001: Two-stage hierarchy (cheap UDP stage, escalate to UDP+QUIC)

Status: accepted

## Context

Every UDP flow carries the 114 UDPFlowLyzer statistics; the 18 QUIC features only
mean anything for QUIC-carrying flows and cost more to extract. A single model on
the joint space pays the full feature cost on every flow.

## Decision

Stage 1 is a small forest on the UDP-only features. Flows it classifies with
confidence below 0.90 escalate to Stage 2, a larger forest on the joint UDP+QUIC
space. Measured on the held-out split: Stage 1 answers 75.7% of flows alone.

## Honest accounting

The ablation in `train.py` shows a single 200-tree forest on the joint space ties
the hierarchy on full-coverage accuracy and macro-F1 (0.8317 / 0.3911, both). The
hierarchy does not buy accuracy on this dataset, and the README says so. What it
buys, measured: ~76% of flows are answered from features that are always present
and cheap, and the escalation confidence gives the reject option a second,
structurally different signal to work with. This mirrors the finding in the
SECRYPT 2026 paper the repo operationalizes.

## Alternatives rejected

- Single joint-space model: simpler, same accuracy, but pays QUIC extraction on
  every flow and loses the cheap-path story. Kept as the permanent ablation so
  the comparison stays measured.
- Per-protocol router (separate UDP and QUIC models, route by flow type):
  invents a routing rule the dataset does not label cleanly; the QUIC features
  already encode "no QUIC observed" as zeros.

## Consequences

Two models to train and version instead of one; escalation rate (24.3%) is a
monitored number in metrics.json, because a drifting escalation rate is an early
sign the Stage-1 confidence distribution moved.
