# Changelog

There are no tagged releases yet; the package version in `pyproject.toml` is 0.2.0. Entries below
are grouped by commit date and name the commits, so every line can be checked with `git show`.
No reported number changed in any entry unless the entry says it was re-measured.

## Unreleased (2026-10-16)

- README: a reviewer summary at the top (headline number, the command that reproduces it, paper
  DOI, dataset page), one Limitations section that gathers the caveats already stated further down,
  and a Citation section with BibTeX. The SECRYPT 2026 paper is now described as published, with
  its DOI (10.5220/0015033600004103); the README and demo page still said "accepted".
- `CITATION.cff`: cite the software, the paper (preferred citation) and the dataset. Validated with
  `cffconvert --validate`. `tests/test_citation.py` keeps its version, DOI and dataset page in
  step with `pyproject.toml` and the README.
- `model.py` and `data.py`: full type hints and docstrings on the public functions, now enforced by
  mypy (`disallow_untyped_defs`). No behaviour change: `scripts/reproduce.py` still regenerates
  `artifacts/metrics.json` byte for byte.
- This changelog.

## 2026-09-30

Live demo, deployment run in CI, experiment tracking, load test. Main commits:

- Live reject-knob and zero-day page on GitHub Pages; CI retrains, rebuilds the per-flow export
  byte for byte and drives the page in headless Chromium before Pages may deploy (`c7e34c5`).
- Kubernetes manifests and Terraform applied to a kind cluster in CI, load-tested with k6, rolled
  under load; a preStop drain fixed the one failed request seen on a rolling restart, checked in a
  24-run A/B (`DEPLOY.md`).
- MLflow tracking of the four model arms and an Evidently drift and performance report, both
  checked against the committed artifacts (`7ab277d`, `d5a0367`). XGBoost is not bit-identical
  across operating systems, so that job runs on Windows (`f82fa64`).
- HTTP load test under concurrent clients found a GIL convoy; scoring now runs under a
  per-process lock with an off switch for the A/B (`1ae761f`, `3e1b02e`, `1299cd8`).
- Zero-day experiment: a rarity control added and the novelty claim withdrawn (`3da639c`).

## 2026-08-05

- Leave-one-family-out experiment: what the model does with an attack family it never saw, with a
  shuffled-label control (`97e2c04`, `739ea4e`).

## 2026-07-20 to 2026-07-23

- Per-family precision and recall, and where the missed families go (`79c17d7`, `9090ae1`).
- Calibration measured on the shipped model's confidence (`a9e4360`).
- LightGBM and XGBoost comparison arms, with SHAP attribution for the winner (`33ef008`,
  `d7196ba`).
- CI status badge (`eb683f3`).

## 2026-07-16

- Scoring benchmark committed; removing a per-call thread-pool spin-up took a single flow from
  42.7 ms to 2.4 ms (`4baf46d`, `8a850bc`).
- Config via pydantic-settings, estimator registry, alert sinks, PSI drift, liveness and readiness
  endpoints, a bounded batch endpoint.
- Eight ADRs and the architecture map (`fdca12f`).
- Locked environment and the byte-identity reproducibility contract; mypy clean (`57c2d9d`).
- Hierarchy ablation: the two-stage design measured against single models, and the unflattering
  result reported (`df3b4bc`).

## 2026-07-15

- Rebuilt on the real BCCC-UDP-QUIC-IDS-2025 dataset; version 0.2.0 (`a2da13c`).

## 2026-07-14

- First version: hierarchical IDS with a tunable reject option (`3e2f1b8`).
