# ADR 007: Sequential tree scoring for small batches

Status: accepted

## Context

The serving path measured ~23 flows/s. Profiling (see the bench commit) showed
the model was innocent: sklearn's forest predict_proba spins up and tears down a
joblib thread pool on every call, ~30-60 ms of overhead per single-row request,
against ~0.3 ms of actual imputation and ~2 ms of tree work.

## Decision

`model.forest_proba(sequential=True)` walks the fitted trees in order with the
per-call validation hoisted out of the loop. It is the same accumulate-then-
divide computation sklearn runs with n_jobs=1, and a test asserts bit-identical
output (np.array_equal, not allclose); if that test ever fails, the fast path is
deleted, not tuned. FlowScorer picks the strategy by batch size: sequential wins
below ~1k rows, the native threaded path wins above ~4k (measured; cutoff 2048).
Non-forest estimators fall back to their native path.

Measured result (artifacts/benchmark.json): single flow mean 42.7 ms to 2.4 ms
(p95 5.6 ms), full-sample batch ~125k flows/s. The pre-fix path stays in the
benchmark as `single_row_native_pool` so the claim remains reproducible.

## Alternatives rejected

- ONNX export / treelite compilation: real speed, but a second model format to
  version and a conversion step to distrust, to beat numbers that are already
  two orders of magnitude past the need.
- A batching queue in front of the service (accumulate requests, score together):
  adds latency to the common case and infrastructure this repo does not need;
  /predict/batch gives callers the batch path explicitly.
- Setting n_jobs=1 at fit time: fixes serving by slowing training, and batch
  scoring genuinely benefits from the threaded path at scale.

## Consequences

Roughly 30 lines in model.py mirror what sklearn does internally, pinned by an
exact-equality test and a perf regression test. If sklearn changes its
accumulation semantics, the equality test catches it on the spot.
