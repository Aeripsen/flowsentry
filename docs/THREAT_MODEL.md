# Threat model: FlowSentry

Short and specific to a flow-based IDS. The system under consideration: flow features -> two-stage
classifier with a reject option -> FastAPI service. Assets worth attacking: the verdict on a flow
(the attacker wants "normal" or at least "unknown-and-ignored"), the model's decision boundary,
and future training data. This document will grow with the Week 3 adversarial probe; the scenarios
below are the ones that probe is designed around.

## Adversary scenarios

### 1. Evasion (feature-space perturbation)

The attacker perturbs what they control to move a malicious flow across the decision boundary.
For flow features the attacker-controllable surface is asymmetric:

| Feature group | Attacker control |
|---|---|
| packet sizes, byte counts (`src_bytes`, `dst_bytes`) | high (padding) |
| timing, duration, rate | high (rate shaping, jitter) |
| flags, protocol, service | medium (constrained by the attack actually working) |
| host-level aggregates (`count`, `srv_count`, `dst_host_*` rates) | low alone; reducible by going slow-and-low across many sources |

Working hypothesis, to be measured, not assumed: a perturbed flow that leaves the attack's learned
region tends to land between class modes, where confidence drops, so the reject option turns a
would-be confident misclassification into an abstention that can be routed to review. The two-stage
design also means evading the cheap stage 1 just buys the attacker an evaluation by the stronger
stage 2. **Week 3 probe:** apply bounded perturbations to held-out attack flows on the
attacker-controllable features only, and report three measured rates: still-detected, abstained,
and successfully evaded (published whatever they turn out to be).

### 2. Mimicry

Instead of perturbing away from the attack region, the attacker shapes the flow to sit inside the
benign manifold: a slow-rate DoS that statistically resembles normal web traffic, or an r2l attempt
inside an otherwise ordinary session. This is the hard case for any statistical flow detector, and
honesty requires saying so: a sufficiently good mimic gets a confident `normal` and the reject
option does not fire, because the model is not uncertain, it is confidently wrong. Partial
mitigations: mimicry imposes a real cost on the attacker (throttling the attack to look benign
blunts it); drift monitoring (Week 3) can surface a slow shift in the benign distribution; and a
flow classifier should be one layer, not the only detector.

### 3. Poisoning

If a future retraining loop ingests production flows with automatic or operator labels, an attacker
who can get flows into that pipeline can shift the boundary (label flipping, or seeding benign-looking
attack flows labeled normal). Current posture: there is no live retraining, the model trains only on
the static public benchmark, so there is no poisoning surface today. Rules for when retraining lands
(Week 2/3): treat auto-labels as untrusted input, require review for samples near the decision
boundary, and keep a frozen canary evaluation set so per-class degradation after a retrain is
detected instead of silently absorbed.

### 4. Oracle abuse of the API

`/predict` returns a confidence score. An attacker with unlimited queries can map the decision
boundary and craft evasions offline, or approximate the model outright (extraction). This is
acceptable for a local benchmark artifact and not acceptable for the public deploy, so the Week 3
deploy carries: authentication and rate limiting on `/predict`, coarse confidence buckets for
untrusted callers, and logging of query patterns consistent with boundary probing (many
near-duplicate flows with small perturbations).

### 5. The service itself

Input is validated by pydantic (typed fields, bounded `reject_threshold`); unknown feature keys are
ignored and unseen categorical values are absorbed by the encoder rather than crashing the encoder.
Volumetric DoS against the endpoint is handled at the deployment layer (rate limits, container
resource caps), not in the model.

## What the reject option is, and is not

It is an uncertainty control: it converts low-confidence outputs, including many off-manifold
perturbed flows, into explicit abstentions with a measured coverage cost (see the curve in the
README). It is not an adversarial-robustness guarantee: mimicry beats it by design, and its actual
behavior under directed perturbation is exactly what the Week 3 probe exists to measure.
