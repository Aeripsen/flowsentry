# ADR 003: PR-AUC as the headline metric

Status: accepted

## Context

The full BCCC-UDP-QUIC dataset is ~94% one flood family; even the balanced
committed sample has 200-400 flows for each rare family against thousands of
benign/UDP-RAW rows. Under imbalance, accuracy and ROC-AUC read high while the
model is useless on the minority classes that matter.

## Decision

The headline is binary attack-detection PR-AUC (average precision), 0.9767 on
the held-out grouped split, plus per-class PR-AUC and F1 for all 8 classes, the
ugly ones included (UDP-OVH at 0.045 is in the README table). Accuracy and
macro-F1 are reported alongside, labeled full-coverage, so nothing is hidden.

## Alternatives rejected

- Accuracy: 83.2% here is mostly "can you tell UDP-RAW from benign"; it says
  nothing about the rare families and rewards majority-class bias.
- ROC-AUC: optimistic under imbalance because the false-positive rate divides by
  the huge negative count; precision-recall reflects what an analyst actually
  sees in an alert queue.
- Reporting only the accepted-subset numbers (coverage ~0.78, reliability ~0.99):
  that is the reject option working, not overall skill, and quoting it alone
  would be the exact overclaim this repo exists to avoid.

## Consequences

The README table contains numbers that look bad (rare-family PR-AUC below 0.1).
That is deliberate: they are real, they motivate the reject option, and removing
them would trade credibility for cosmetics.
