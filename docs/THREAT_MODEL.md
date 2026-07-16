# Threat model: FlowSentry

Short and specific to a flow-based IDS. The system under consideration: UDP/QUIC flow features ->
two-stage classifier with a reject option -> FastAPI service. Assets worth attacking: the verdict on
a flow (the attacker wants `benign`, or at least `unknown-and-ignored`), the model's decision
boundary, and future training data. This document will grow with an adversarial probe; the scenarios
below are the ones that probe is designed around.

## Adversary scenarios

### 1. Evasion (feature-space perturbation)

The attacker perturbs what they control to move a malicious flow across the decision boundary. For
UDP/QUIC flow features the attacker-controllable surface is asymmetric:

| Feature group | Attacker control |
|---|---|
| packet sizes, byte counts (`avg_pkt_size`, `byte_count`, `pkt_size_*`) | high (padding) |
| timing, duration, rates (`mean_iat`, `iat_std`, `pps`, `bps`, burst structure) | high (rate shaping, jitter) |
| directional stats (`directional_asymmetry`, `fwd_bwd_*`) | medium (constrained by the attack actually working) |
| QUIC handshake/path signals (`quic_used0rtt_any`, `quic_migrations_sum`, ...) | low (dictated by a real QUIC stack) |

Working hypothesis, to be measured, not assumed: a perturbed flow that leaves the attack's learned
region tends to land between class modes, where confidence drops, so the reject option turns a
would-be confident misclassification into an abstention that can be routed to review. The two-stage
design also means evading the cheap Stage 1 (UDP-only) just buys the attacker an evaluation by the
stronger QUIC-augmented Stage 2. **Probe:** apply bounded perturbations to held-out attack flows on
the attacker-controllable features only, and report three measured rates: still-detected, abstained,
and successfully evaded (published whatever they turn out to be).

### 2. Mimicry

Instead of perturbing away from the attack region, the attacker shapes the flow to sit inside the
benign manifold: a low-rate flood whose statistics resemble ordinary UDP application traffic. This
is the hard case for any statistical flow detector, and honesty requires saying so: a sufficiently
good mimic gets a confident `benign` and the reject option does not fire, because the model is not
uncertain, it is confidently wrong. Partial mitigations: mimicry imposes a real cost on the attacker
(throttling a flood to look benign blunts it); drift monitoring can surface a slow shift in the
benign distribution; and a flow classifier should be one layer, not the only detector.

### 3. Poisoning

If a future retraining loop ingests production flows with automatic or operator labels, an attacker
who can get flows into that pipeline can shift the boundary (label flipping, or seeding
benign-looking attack flows labeled benign). Current posture: there is no live retraining, the model
trains only on the static public dataset sample, so there is no poisoning surface today. Rules for
when retraining lands: treat auto-labels as untrusted input, require review for samples near the
decision boundary, and keep a frozen canary evaluation set so per-class degradation after a retrain
is detected instead of silently absorbed.

### 4. Oracle abuse of the API

`/predict` returns a confidence score. An attacker with unlimited queries can map the decision
boundary and craft evasions offline, or approximate the model outright (extraction). This is
acceptable for a local research artifact and not acceptable for a public deploy, so the deploy
carries: authentication and rate limiting on `/predict`, coarse confidence buckets for untrusted
callers, and logging of query patterns consistent with boundary probing (many near-duplicate flows
with small perturbations).

### 5. The service itself

Input is validated by pydantic at the boundary: feature values must be finite numbers (strings and
NaN/Infinity literals are rejected with 422, not fed into the feature row), `reject_threshold` is
bounded to [0, 1], and `/predict/batch` is capped at `max_batch_rows` (default 4096) per request,
which bounds both memory and per-request oracle throughput. Unknown feature keys are ignored,
missing UDP features are median-imputed and missing QUIC features default to 0, rather than crashing
the pipeline. `/health` is liveness-only and never touches the model; `/ready` owns the 503.
Volumetric DoS against the endpoint is handled at the deployment layer (rate limits, container
resource caps), not in the model.

### 6. The model artifact (pickle trust boundary)

The serving artifact (`artifacts/flowsentry.joblib`) is joblib, which is pickle: loading one
executes whatever it contains. The rule, enforced by practice and documented in
`scoring.load_bundle`, is that the service only ever loads an artifact this repo trained itself,
locally via `python -m flowsentry.train` or inside the Docker build (the image trains from the
committed sample at build time, so nothing is downloaded at runtime). No artifact is fetched over
the network, accepted from a user, or shipped in the repo (`artifacts/*.joblib` is gitignored). If a
future deploy wants prebuilt artifacts, they need integrity verification (signed digests) before
`load` ever sees them; until then, do not point `load_bundle` at anything you did not train.

## What the reject option is, and is not

It is an uncertainty control: it converts low-confidence outputs, including many off-manifold
perturbed flows, into explicit abstentions with a measured coverage cost (see the curve in the
README). It is not an adversarial-robustness guarantee: mimicry beats it by design, and its actual
behavior under directed perturbation is exactly what the adversarial probe exists to measure.
