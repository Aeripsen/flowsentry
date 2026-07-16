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

On the committed balanced sample the grouping changes nothing measurable: the
sample averages ~1.4 flows per connection (the row-capping that balances classes
also dilutes connections), and a plain stratified split scores the same binary
PR-AUC (0.9767). The grouping is kept because it is the correct method and it
bites on the full dataset (~25 flows per 5-tuple for the dominant flood). The
deeper limit is stated in the model card: UDP-RAW comes from 2 source IPs, so
even grouped, the same attacking host straddles the split on different ports.
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
