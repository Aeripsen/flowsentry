# ADR 006: What is config and what is schema

Status: accepted

## Context

"No magic constants" taken literally would move the feature lists, the class
set, and every threshold into YAML. Some of those are tunables; some are the
contract that makes the published numbers mean anything.

## Decision

Config (pydantic-settings, env > optional YAML > code defaults; see config.py):
split size, seed, escalation threshold, reject-threshold grid, stage estimator
name + params, artifact/sample paths, serving cutoffs and caps. The in-code
defaults are the exact measured configuration and a test pins each one.

Schema (code, guarded by tests, not configurable): the 114 UDP and 18 QUIC
feature names, their order, the class set, the connection-key columns. These are
the dataset contract. A config knob for them would let a typo train a silently
different model whose numbers still get quoted as if they were the published
ones.

Two guardrails in the config layer itself: `extra="forbid"` so an unknown key
fails the run instead of being ignored, and the defaults-pinning test so nobody
drifts the published configuration by editing a fallback.

## Alternatives rejected

- Everything in YAML: maximizes flexibility, destroys the link between the repo
  and its published numbers.
- Everything hardcoded: retraining with a different estimator or seed becomes a
  code edit, which is friction exactly where experimentation is legitimate.

## Consequences

Changing the feature schema requires a code change plus updating the tests that
pin it. That is the point.
