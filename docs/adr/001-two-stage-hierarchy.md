# ADR 001: Two-stage hierarchy (cheap UDP stage, escalate to UDP+QUIC)

Status: accepted (the architecture stays, because implementing the paper is this
repo's job). The compute justification this ADR used to make is **withdrawn**: it
was never measured, and now that it has been, it does not hold.

## Context

Every UDP flow carries the 114 UDPFlowLyzer statistics; the 18 QUIC features only
mean anything for QUIC-carrying flows and cost more to extract. A single model on
the joint space pays the full feature cost on every flow.

## Decision

Stage 1 is a small forest on the UDP-only features. Flows it classifies with
confidence below 0.90 escalate to Stage 2, a larger forest on the joint UDP+QUIC
space. Measured on the held-out split: Stage 1 answers 75.7% of flows alone.

## Honest accounting: the hierarchy does not pay for itself on this sample

This ADR used to say the hierarchy buys compute, because ~76% of flows are answered
from cheap always-present features and never touch the 200-tree model. That was the
repo's central architectural claim and it was the only one with no artifact behind
it. `scripts/hierarchy_benchmark.py` measures it now and writes
`artifacts/hierarchy_benchmark.json`. The claim does not survive contact with the
measurement.

Four arms, same trained artifact, same held-out rows (quality is seeded and exact;
latency is one run's snapshot, see the artifact's stability note):

| Arm | Accuracy | Macro-F1 | Binary PR-AUC | Serving ms/flow | Batch ms/flow |
|---|---|---|---|---|---|
| stage1_only (60 trees, 114 UDP) | 0.8306 | 0.3794 | 0.9764 | 1.595 | 0.00769 |
| single_joint (200 trees, 132) | 0.8317 | 0.3911 | **0.9774** | 5.010 | 0.01796 |
| single_joint_small (60 trees, 132) | 0.8311 | **0.3946** | 0.9771 | **1.388** | **0.00704** |
| hierarchy (shipped) | 0.8317 | 0.3911 | 0.9767 | 2.597 | 0.02453 |

Read the third row against the fourth. A single 60-tree forest on the joint space is
faster than the hierarchy on both paths (1.9x on the serving path, 3.5x in batch),
scores the **highest macro-F1 of any arm**, and beats the hierarchy on binary PR-AUC.
It is one model instead of two, with no escalation threshold to tune.

The hierarchy only wins against the 200-tree joint model, which is the baseline this
ADR happened to pick. That is not a defence. The 60-tree joint model is the baseline
that should have been tried, and nobody tried it.

### The reject knob does not rescue it

The obvious fallback is that the hierarchy exists to serve the reject option. It does
not need to: any model that emits a probability has a reject knob. The measured
coverage-reliability curves of all four arms sit within noise of each other
(`reject_curves` in the artifact), and at the strictest setting, threshold 0.99, both
joint models **dominate** the hierarchy outright:

| Arm | Coverage @ 0.99 | Reliability @ 0.99 |
|---|---|---|
| hierarchy | 0.6476 | 0.9932 |
| single_joint | **0.6662** | **0.9941** |
| single_joint_small | **0.6536** | **0.9942** |

More coverage and more reliability at the same time, from one model. The reject knob
is a good idea and the curve is still the product; it just is not an argument for two
stages.

### Why the accuracy tie is exact rather than close

`train.py` fits Stage 2 on all training rows with the same config and seed as the
ablation model, so `model.stage2_` and `ablation_single_rf` are the same fitted
forest. The hierarchy is therefore "Stage 1 where Stage 1 is confident, the joint
model everywhere else", and the artifact confirms it: on the escalated rows the
hierarchy and the joint model agree on **100.0%** of labels
(`hierarchy_agrees_with_joint_on_escalated_rows`). The tie is structural, not a
coincidence, and not a bug.

### What the hierarchy is actually for

Three things survive. None of them is speed and none of them is accuracy.

1. **It is the paper's architecture.** This repo exists to operationalize the
   SECRYPT 2026 hierarchical UDP/QUIC design on that paper's own dataset.
   Implementing the published method faithfully and then measuring that it does not
   beat a simpler baseline **on this sample** is the honest result, and it is worth
   more than a repo that quietly swapped in a single forest and said nothing.
2. **The escalation rate is a monitoring signal.** It ships in `metrics.json`
   (0.2429), and a drifting escalation rate is an early sign the Stage-1 confidence
   distribution moved. That is one real number a single model does not give you.
3. **A deferred-extraction deployment could still pay.** This is where the original
   argument might genuinely live. A pipeline that ran UDPFlowLyzer, scored Stage 1,
   and only called QUICFlowLyzer on the 24.3% that escalate would skip QUIC
   extraction for three flows in four. The artifact prices that honestly instead of
   inventing a number for it. Writing `C_quic` for the per-flow QUIC extraction cost
   and `e` for the escalation rate:

       advantage = (1 - e) * C_quic + [T_baseline - T_stage1 - e * T_stage2]

   The extraction term cannot be negative, so if the measured bracket were positive
   the hierarchy would win for any `C_quic` at all. Against the 200-tree baseline it
   is positive. Against the 60-tree joint model it is **negative**, so `C_quic` has to
   clear roughly **1.9 ms per flow** on the serving path before the hierarchy pays for
   itself. This repo cannot measure `C_quic`: the extractors are upstream tools living
   in other repos, and every benchmark here starts from already-computed features
   (ADR 008). So this stays an open question with a stated break-even, not a claim.

   **And FlowSentry does not implement it anyway.** `scoring.row_from_features` builds
   all 132 columns, QUIC slots included, before Stage 1 ever runs, so the served path
   pays for whatever the caller computed regardless of escalation. The deferred
   deployment is what the architecture permits, not what this code does.

## Alternatives rejected

- **Single joint-space model: no longer rejected on the evidence.** A 60-tree joint
  forest is simpler and faster, scores the highest macro-F1 of any arm, and beats the
  hierarchy on binary PR-AUC, at a cost of 0.0006 lower accuracy (0.8311 vs 0.8317). It is
  kept as the permanent ablation and as the benchmark's control arm. The hierarchy
  stays because the repo's job is to implement the paper, and this ADR says plainly
  what that costs rather than inventing a benefit.
- Per-protocol router (separate UDP and QUIC models, route by flow type): invents a
  routing rule the dataset does not label cleanly; the QUIC features already encode
  "no QUIC observed" as zeros.

## Consequences

Two models to train and version instead of one, for no measured accuracy or compute
gain on this sample. The escalation rate (24.3%) stays a monitored number in
metrics.json. If this were a product rather than a faithful implementation of a
published architecture, the 60-tree joint model is what should ship, and this ADR
would be superseded rather than annotated.

Scope, stated so this is not over-read: this is a result about the committed 25,615-flow
balanced sample, with this feature set and this estimator. It is not a claim about the
paper's own evaluation, which uses a different sample, different preprocessing and a
different protocol. Two numbers computed on different protocols are not a scoreboard.
