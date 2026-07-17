# ADR 002: Connection-grouped train/test split

Status: accepted

## Context

Flow datasets are not i.i.d.: flows from the same 5-tuple connection are highly
correlated. A random row split lets near-duplicates of a training flow land in
the test set, which inflates scores without the model generalizing.

## Decision

`GroupShuffleSplit` on the UDP 5-tuple (src_ip, src_port, dst_ip, dst_port), so
no connection straddles the split. The median imputer fits on train only. A test
asserts the disjoint-connections property on every run. IPs and ports are used
only to build the group key, never as features.

## Honest accounting

Measured, not asserted: `scripts/split_comparison.py` runs both splits head to
head at the same seed and test_size and writes artifacts/split_comparison.json.
On the committed balanced sample the grouping changes nothing measurable. Binary
PR-AUC is identical (0.9767 either way), accuracy differs by four ten-thousandths
(grouped 0.8317, stratified 0.8321), and macro-F1 is actually *higher* under the
grouped split (0.3911 vs 0.3714).

The mechanism is the sample, not the method. It averages 1.4003 flows per
connection, and the dominant flood averages 1.283 (12,000 flows over 9,353
connections). The row-capping in `scripts/build_sample.py` samples the two
dominant classes down at random, which scatters their flows across connections,
so grouping has almost nothing left to hold together. It is kept because it is
the correct method and because the correlation it guards against is a real
property of the full dataset, which this repo does not ship and therefore does
not quote a number for.

The deeper limit it does NOT fix, from the same artifact: UDP-RAW comes from 2
source IPs, and 85.24% of test flows share a source IP with training. So even
grouped, the same attacking host straddles the split on different ports.
Host-level generalization is untestable inside this dataset.

## Alternatives rejected

- Random stratified split: measurably equivalent here, wrong in general, and the
  first thing a reviewer checks in an IDS repo.
- Grouping by source IP: the honest next level, but with 2 attack source IPs it
  collapses the attack classes into one side of the split; needs a cross-day or
  cross-dataset eval instead (roadmap, stated in the model card).

## Consequences

Slightly fewer effective test configurations than a row split; the leakage guard
test (`test_split_is_connection_leakage_safe`) is load-bearing and must not be
weakened.
