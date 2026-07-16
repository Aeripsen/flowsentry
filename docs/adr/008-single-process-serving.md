# ADR 008: Single-process serving, and the infrastructure deliberately not built

Status: accepted

## Context

An IDS repo invites architecture: queues for the flows, a database for the
alerts, microservices for the stages, Kubernetes for the microservices. Each of
those is justified by a load profile this repo does not have and cannot honestly
claim: it scores stored flows and serves a research model.

## Decision

One FastAPI process wrapping one FlowScorer. Batch throughput needs are met by
the vectorized path (measured ~125k flows/s on the dev machine), which is orders
of magnitude past the committed dataset's needs. Alerts go to sinks (stdout,
JSONL); persistent storage of alerts is the consumer's job (a SIEM tails the
JSONL), not this service's.

## Deliberately not built, and why

- Message queue / stream processor (Kafka and friends): there is no live event
  source in this repo; a queue in front of a CSV replay is theater.
- Alert database: the JSONL sink is the integration point; owning storage means
  owning retention, schema migration, and querying, all for data a SIEM already
  handles better.
- Microservice split (stage 1 and stage 2 as separate services): adds a network
  hop inside a 2 ms code path.
- Kubernetes manifests: one container with a health/readiness split is exactly
  as deployable as this needs to be; docker-compose covers the two-process case
  (API + dashboard).
- A plugin system for sinks/models beyond the registry: the registry and the
  sink protocol each have two real implementations; a discovery mechanism for
  hypothetical third parties has zero.

## Consequences

If a live tap ever feeds this system, the first real bottleneck will be feature
extraction upstream of the model, not scoring, and that is where design effort
should go at that point. Horizontal scale, if needed, is "run more containers
behind a load balancer": the service is stateless.
