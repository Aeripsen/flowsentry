# ADR 005: What ships in git, what gets built, what gets loaded

Status: accepted

## Context

The repo must reproduce from a cold clone without a multi-GB download, without
shipping a 65 MB model binary in git, and without ever loading a pickle anyone
else produced.

## Decision

- Data: a 25,615-flow stratified sample of the public CC BY 4.0 dataset is
  committed (9.4 MB gzip). All rare-family flows kept, the two dominant classes
  capped, rebuildable from the full dataset via `scripts/build_sample.py`.
- Model: never committed (`artifacts/*.joblib` is gitignored). `python -m
  flowsentry.train` rebuilds it deterministically in ~30 s, and the Docker image
  trains at build time from the committed sample, so a clean clone always gets a
  working /predict without downloading an artifact from anywhere.
- Format: joblib, loaded only from the local artifacts directory. joblib is
  pickle, so the trust boundary is "only load what this repo trained"; that rule
  lives in `scoring.load_bundle` and docs/THREAT_MODEL.md section 6.
- Measured evidence: metrics.json, benchmark.json, calibration_experiment.json,
  split_comparison.json, hierarchy_benchmark.json ARE committed. They are small,
  diffable, and `make reproduce` checks the retrained metrics.json is
  byte-identical to the committed one. The rule they enforce: a number that appears
  in the README, an ADR, the model card or a code comment cites one of these files,
  or it does not ship. That rule cuts both ways: hierarchy_benchmark.json is
  committed even though its verdict is that the repo's own headline architecture
  does not beat a simpler baseline.

## Alternatives rejected

- Committing the trained model (or via git-lfs): a binary nobody can review, a
  pickle the clone is asked to trust, and drift between code and artifact.
- Downloading the model at container start: adds a network dependency and an
  integrity problem (signing infrastructure) to save 30 s of build time.
- ONNX as the serving format: solves pickle distrust, but this repo does not
  distribute artifacts at all, so it buys nothing today; noted as the right move
  if prebuilt artifacts ever ship.

## Consequences

Docker builds take the training time. Anyone who wants the full-dataset numbers
must fetch the full dataset themselves (scripts/get_data.py explains).
