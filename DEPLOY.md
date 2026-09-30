# Deploying FlowSentry

The serving image (`Dockerfile`) trains its model at build time from the committed
BCCC-UDP-QUIC sample (25,615 flows), so a build from a clean clone serves a model
trained on real data with no download. It installs dependencies from the
`pyproject.toml` ranges, not `requirements.lock`, so it is the same training recipe
as the committed metrics but not guaranteed byte-identical to that model.

The service listens on port 8000, runs as uid 10001, and exposes:

- `GET /health` - liveness, always 200 while the process is up; never touches the model
- `GET /ready` - readiness, 503 until a trained model is loaded
- `POST /predict`, `POST /predict/batch`, `GET /curve`

## What has actually run

| Target | Status | Evidence |
|---|---|---|
| Kubernetes (kind) | Deployed, smoke-tested, load-tested, PodDisruptionBudget exercised and rolled under load in CI on every deploy change | [`.github/workflows/k8s.yml`](.github/workflows/k8s.yml), [`artifacts/k8s_kind_flowsentry.json`](artifacts/k8s_kind_flowsentry.json) |
| Connection drain, with vs without | 24 runs on fresh runners, 12 each way | [`.github/workflows/k8s-drain-ab.yml`](.github/workflows/k8s-drain-ab.yml), [`artifacts/k8s_drain_ab_flowsentry.json`](artifacts/k8s_drain_ab_flowsentry.json) |
| Terraform, `hashicorp/kubernetes` | Applied to a throwaway kind cluster, re-planned, diffed against the YAML, destroyed, in CI | [`artifacts/terraform_kind_flowsentry.json`](artifacts/terraform_kind_flowsentry.json) |
| docker compose | Brought up and smoke-tested in CI on every deploy change (API `/health`, `/ready`, `/predict`, dashboard health) | the `compose` job in `k8s.yml`, [`scripts/compose_smoke.sh`](scripts/compose_smoke.sh) |
| Manifest schemas | `kubeconform -strict` in CI on every deploy change | the `kubeconform` job, [`scripts/k8s_schema.sh`](scripts/k8s_schema.sh) |
| Any cloud | **Not deployed** | no cloud account is used |

Everything in CI runs on a free GitHub-hosted runner (4 vCPU, 16 GB). "Applied"
for Terraform means applied to a kind cluster that exists for the length of one CI
job, not to cloud infrastructure.

---

## 1. Kubernetes

Layout:

- [`deploy/k8s/base/flowsentry.yaml`](deploy/k8s/base/flowsentry.yaml): ConfigMap
  (`FLOWSENTRY_SERVING__*` settings, injected with `envFrom`), Deployment (2 replicas,
  non-root uid 10001, liveness `/health`, readiness `/ready`, `maxUnavailable: 0`
  rolling update, preStop drain, requests 250m CPU / 384Mi, limits 1 CPU / 1Gi),
  ClusterIP Service, HorizontalPodAutoscaler (CPU 70% of request, 2 to 5 replicas)
  and a PodDisruptionBudget (`minAvailable: 1`).
- [`deploy/k8s/overlays/kind`](deploy/k8s/overlays/kind): the base in its own
  namespace, with the locally built `flowsentry:ci` image and one ConfigMap value
  changed (`FLOWSENTRY_SERVING__MAX_BATCH_ROWS=2000`) so the run can prove the
  ConfigMap reaches the process.
- [`deploy/k8s/loadtest`](deploy/k8s/loadtest): a k6 Job that runs inside the
  cluster and posts one flow to `/predict` through the Service.

### Reproduce it

Needs docker, [kind](https://kind.sigs.k8s.io/) and kubectl. No account.

```bash
make k8s-e2e          # or: bash scripts/k8s_e2e.sh
make k8s-schema       # kubeconform -strict on everything below
make compose-smoke    # docker compose up, smoke test, down
```

`scripts/k8s_e2e.sh` is exactly what CI runs:

1. `docker build -t flowsentry:ci .` (trains the model), `kind create cluster`,
   `kind load docker-image`
2. install metrics-server (the HPA needs it), `kubectl apply -k deploy/k8s/overlays/kind`,
   `kubectl rollout status` (returns once every pod passed `/ready`)
3. through the Service: `/health`, `/ready`, `/predict`, and `/openapi.json`, whose
   `/predict/batch` `maxItems` must equal the ConfigMap value; record PID 1 and the uid
4. PodDisruptionBudget: with 2 pods up, evict one through the Eviction API (must be
   allowed), then the other straight away (must be refused)
5. steady load: 16 k6 virtual users for 90 s, closed loop, while logging the HPA;
   then count the requests each pod served from its uvicorn access log
6. rolling restart under load: a new 120 s k6 run, `kubectl rollout restart` 30 s
   into it; per-pod counts again just before the restart and, for the replacement
   pods, at the end. k6 counts requests and failures per 10 s of wall clock, so the
   report can compare the 30 s before the restart with the windows during and after
7. [`scripts/k8s_report.py`](scripts/k8s_report.py) writes the artifact and **fails
   the job on any failed request**, an unreplaced pod, a rollout that outlasted the
   load, a ConfigMap value that did not reach the process, a shell as PID 1, or a
   PodDisruptionBudget that did not refuse the second eviction

To deploy the base to a real cluster instead, push the image somewhere the cluster
can pull, set `image:` and run `kubectl apply -k deploy/k8s/base`.

### Captured output

From CI run [36777986130](https://github.com/Aeripsen/flowsentry/actions/runs/36777986130)
on master (commit `425ff88`), kind v0.33.0, Kubernetes v1.37.0. The full JSON is
[`artifacts/k8s_kind_flowsentry.json`](artifacts/k8s_kind_flowsentry.json).

```text
rollout: 2 pods ready in 6 s
PID 1:   /usr/local/bin/python3.12 /usr/local/bin/uvicorn flowsentry.service:app ...
runs as: uid=10001(app)
ConfigMap FLOWSENTRY_SERVING__MAX_BATCH_ROWS=2000, served /predict/batch maxItems=2000
/predict: {"label": "unknown", "confidence": 0.5251, "escalated_to_stage2": true, "abstained": true}
PDB:     first eviction allowed; second refused: "Cannot evict pod as it would
         violate the pod's disruption budget."

steady (16 VUs, 90 s): 15,904 requests, 0 failed
  before the scale-out 174.7 req/s, with all 5 pods running 176.8 req/s
  requests per pod: 8,026 / 7,878 / 0 / 0 / 0
rolling restart (120 s): 39,199 requests, 0 failed
  restart began 32 s into the load, took 37 s, all 5 pods replaced
  before the restart 352.8 req/s, during 276.1, after 351.4
  per pod in the 30 s before the restart: 2,803 / 2,718 / 2,697 / 2,276 / 322
  per replacement pod over the phase:    5,880 / 4,795 / 4,770 / 4,590 / 524
HPA: chose 5 replicas 25 s into the load, 5 pods running at 40 s;
  the 2 busy pods sat at their CPU limit (999m of 1000m)
```

The `/predict` answer is for the example flow in the API docs with
`reject_threshold: 0.9`; its confidence is below the threshold, so it escalates to
stage 2 and abstains. That is the reject option working, not an error. It also
means the load test sends that one flow every time, so every request takes the
stage-2, abstain path: the latency and throughput here describe that one input,
not a traffic mix.

### What the runs found

**The first run failed its own gate.** With only a preStop `sleep 5`, a rolling
restart under load failed 1 of 40,070 requests with `connection reset by peer`
([run 36769507432](https://github.com/Aeripsen/flowsentry/actions/runs/36769507432),
[`artifacts/k8s_kind_flowsentry_before_drain.json`](artifacts/k8s_kind_flowsentry_before_drain.json)).
The LedgerSentry deployment, built the same way, failed 4 of 114,308 the same way.

Why: the sleep only covers *new* connections. When a pod starts terminating it
leaves the Service's endpoints and kube-proxy stops sending new connections to it,
but a client's existing keep-alive connection stays pinned to that pod. When
uvicorn gets SIGTERM it closes idle keep-alive connections, and a client that sends
its next request on one at that instant gets a reset.

The fix ([`src/flowsentry/drain.py`](src/flowsentry/drain.py)): preStop runs
`touch /tmp/draining && sleep 5`. While that file exists, a small ASGI middleware
adds `Connection: close` to every response, so each client finishes its current
request, closes the connection cleanly and reconnects to a pod that is staying.
It is a file rather than an endpoint so nothing reachable over the network can put
a pod into drain mode.
[`tests/test_drain_socket.py`](tests/test_drain_socket.py) checks the part a test
client cannot: it starts a real uvicorn server on a local port, once with each of
uvicorn's HTTP parsers (h11 and httptools), and over a raw socket asserts that the
server closes the TCP connection after the drain response, and that without the
drain file the same socket carries a second request. It runs in the normal CI test
job. The image installs `uvicorn[standard]`, which brings httptools, and uvicorn
picks httptools when it is installed.

**Whether the drain works, measured.** Each rolling restart is one event, so the
unit is the run, not the request.
[`k8s-drain-ab.yml`](.github/workflows/k8s-drain-ab.yml) runs the whole of
`k8s_e2e.sh` 12 times on fresh runners: 6 with the drain and 6 with `DRAIN=off`,
which puts the old plain `sleep 5` back. It ran twice, on commits `b24170d`
([run 36776371396](https://github.com/Aeripsen/flowsentry/actions/runs/36776371396))
and `425ff88` ([run 36778001492](https://github.com/Aeripsen/flowsentry/actions/runs/36778001492));
the drain code is the same in both. All 24 runs were valid.

| | runs | runs with a failed request | failed requests per run | requests in the restart phases |
|---|---|---|---|---|
| drain off | 12 | 4 | 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0 | 640,028 |
| drain on | 12 | 0 | all 0 | 665,187 |

A one-sided Fisher exact test on those runs gives p = 0.047
([`artifacts/k8s_drain_ab_flowsentry.json`](artifacts/k8s_drain_ab_flowsentry.json)).
The race is rarer here than in LedgerSentry (at most 1 failed request per run), so
FlowSentry on its own is borderline evidence; LedgerSentry's 9 of 12 against 0 of
12 (p = 0.0002) with the same fix is the stronger result. Before the A/B, the record
was 1 of 1 runs failed without the drain and 0 of 3 with it (40,139, 37,164 and
43,980 requests: attempts 1 and 2 of
[run 36770459936](https://github.com/Aeripsen/flowsentry/actions/runs/36770459936),
which are two attempts of one commit, and the master run
[36771699879](https://github.com/Aeripsen/flowsentry/actions/runs/36771699879);
[`artifacts/k8s_kind_flowsentry_postfix_attempt1.json`](artifacts/k8s_kind_flowsentry_postfix_attempt1.json),
[`_postfix_attempt2.json`](artifacts/k8s_kind_flowsentry_postfix_attempt2.json),
[`_postfix_master.json`](artifacts/k8s_kind_flowsentry_postfix_master.json); attempt 1's
artifact was replaced on GitHub when attempt 2 uploaded, so its report is copied from
the committed job log in [`artifacts/ci_logs/`](artifacts/ci_logs)).

**Keep-alive also defeats the autoscaler's new pods.** Across the 24 A/B runs,
the HPA chose 5 replicas 19 to 40 s into the load and 5 pods were running 34 to 51
s in (the log is sampled every 5 s, so each time is up to 5 s late). But the steady
phase's throughput barely moved: after the scale-out it was 0.98 to 1.64 times the
rate before it, median 1.03. The access logs say why. In 21 of the 24 runs, and in
the master run, 2 of the 5 pods served the steady phase and the other 3 served
nothing (2 requests between them in one run, 0 in the rest; the per-pod log totals match k6's request count in every run). The
16 k6 connections were opened while 2 pods existed and stay on those pods, because
kube-proxy balances connections, not requests. In the other 3 runs a third pod
picked up connections during the steady phase; nothing recorded says which
connection moved or why, so that stays unexplained.

**The rolling restart did not spread the connections.** An earlier version of this
file said it did and credited it with a 2.6x throughput rise. That compared the
steady phase with a whole restart phase whose k6 Job is a *new* client: its
connections open against 5 ready pods before the restart begins. Measured
separately: in the 30 s before the restart, that fresh client was already served by
4 or 5 of the 5 pods (at least 5% of its requests each) in every run where the
counts are complete (23 of 24; in one fast run a pod's log rotated mid-run, which
is why the kind cluster now keeps 200Mi of log per container,
[`deploy/k8s/kind-cluster.yaml`](deploy/k8s/kind-cluster.yaml)), and after the
restart throughput was 0.76 to 1.04 times the pre-restart rate, median 1.00, across
the 24 runs. So a fresh client spreads over the pods that exist when it connects,
and the restart itself adds nothing. Clients that reconnect during the rollout stay
where they land, which can leave a new pod nearly idle (524 of 20,559 requests in
the master run). A real deployment would fix this with an L7 proxy or service mesh
that balances per request. Not done here.

**Per-pod HTTP throughput.** In the master run the 2 busy pods, each at its 1-CPU
limit, served 176.8 req/s between them. That is well below the in-process scoring
benchmark (`python -m flowsentry.bench`). README's load-test section measures the
HTTP service on one laptop and traces part of the gap to a GIL convoy in the
per-tree scoring loop, now scored under a per-process lock; the kind runs from
commit `10a38c8` on include that change. Nothing in the kind runs isolates the rest
of the gap, so no further cause is claimed.

**CPU and memory.** The HPA's `averageUtilization` peaked at 337% of the 250m
request in the master run. It cannot pass 400% here, because the limit is 1000m: a
value near it means the busy pods sat at their CPU limit and were throttled.
Memory was about 265 to 268 Mi per pod under load, against a 384Mi request and a
1Gi limit.

### What these numbers are not

- One single-node kind cluster on one 4-vCPU runner, with the k6 load generator in
  the same cluster, competing for the same CPUs as the pods. The HPA going from 2
  to 5 pods on one node shows the control loop working, not added capacity: 5 pods
  with 1-CPU limits cannot all get their limit on 4 vCPU shared with k6.
- Synthetic load: one fixed flow in a closed loop, which always takes the stage-2
  abstain path. Not production traffic, not a live network tap.
- Throughput varies a lot between runners: steady-phase rates of 105.9 to 176.6
  req/s across the committed runs, and some A/B runners ran more than twice as fast
  as others. The CI gate is therefore on failed requests, never on a throughput
  number, and no req/s or latency figure here is a benchmark.
- The PodDisruptionBudget check proves the eviction API refuses to take the last
  pod. It is not a node drain under load.

---

## 2. Terraform (kubernetes provider, applied to kind in CI)

[`deploy/terraform/kubernetes`](deploy/terraform/kubernetes) creates the same six
objects as the YAML (namespace, ConfigMap, Deployment, Service, HPA, PDB) as typed
`hashicorp/kubernetes` resources. `wait_for_rollout = true` makes `apply` return only
after every pod passed `/ready`, and `ignore_changes` on `replicas` stops Terraform
from fighting the HPA.

```bash
make tf-kind          # or: bash scripts/tf_kind_e2e.sh
```

does `terraform init`, `validate`, `apply` to a kind cluster, a `/ready` and
`/predict` smoke test through the Service, a second `terraform plan
-detailed-exitcode` that must be empty, [`scripts/k8s_parity.py`](scripts/k8s_parity.py)
(the live Terraform objects against a server-side dry run of `deploy/k8s/base`),
then `terraform destroy`.

From the same master run, Terraform 1.16.4, provider 3.2.1
([`artifacts/terraform_kind_flowsentry.json`](artifacts/terraform_kind_flowsentry.json)):

```text
Apply complete! Resources: 6 added, 0 changed, 0 destroyed.      (16 s)
second plan: No changes. Your infrastructure matches the configuration.
parity: 5 objects compared, 0 mismatches
Destroy complete! Resources: 6 destroyed.
```

The parity check compares a chosen contract per object, not every field: for the
Deployment the selector, rollout strategy, pod labels, pod security context,
service links and grace period, and per container the ports, env, probes,
lifecycle, resources and security context; the Service's type, selector and
ports; the full HPA and PDB specs; the ConfigMap data. It skips the image, the
replica count, object metadata labels and the namespace object, and a field set
only on the Terraform side passes if its value is a zero value. "0 mismatches"
means 0 on those fields, not full equivalence.

To use it against another cluster: `terraform -chdir=deploy/terraform/kubernetes
apply -var kube_context=<context> -var image=<registry>/flowsentry:<tag>`.

There is no cloud Terraform module for FlowSentry. Nothing here has been deployed
to a cloud provider.

---

## 3. Local: docker compose

```bash
docker build -t flowsentry . && docker run -p 8000:8000 flowsentry
docker compose up --build       # API on 8000 + dashboard on 8501
```

CI brings this stack up with `docker compose up --wait` on every deploy change
([`scripts/compose_smoke.sh`](scripts/compose_smoke.sh)) and checks the API's
`/health`, `/ready` and `/predict` and the dashboard's `/_stcore/health`. Its first
run found a real bug: the dashboard container inherited the image's `HEALTHCHECK`,
which probes the API on port 8000, so the dashboard was always "unhealthy". It now
checks Streamlit's own health endpoint.

---

## Validation status

- CI, every push that touches a deploy file (`k8s.yml`): the kind deployment, PDB
  check, load test and rolling restart (section 1), the Terraform
  apply/re-plan/parity/destroy (section 2), `terraform fmt -check` plus `validate`,
  `kubeconform -strict` (v0.8.0, its newest published schemas) on the base, the kind
  overlay and the k6 Job, and the docker compose smoke test (section 3).
- CI, every push: `tests/test_deploy_manifests.py` pins the contracts offline:
  selectors, probe paths, named ports, the non-root uid against the Dockerfile,
  `maxUnavailable: 0`, the preStop drain path against `drain.py`, the ConfigMap
  keys against the code defaults, the HPA and PDB targets, and the Terraform
  settings against the YAML ConfigMap. `tests/test_drain_socket.py` runs the drain
  against real uvicorn sockets; `tests/test_k8s_report.py` and
  `tests/test_k8s_drain_ab.py` pin the report arithmetic.
- Manual (`k8s-drain-ab.yml`): the drain A/B above. The committed summary was
  built from both runs' uploaded files, each report regenerated with the current
  `k8s_report.py` so one version of the arithmetic produced every number:

  ```bash
  for run in 36776371396 36778001492; do gh run download $run -p 'ab-*' -D ab/$run; done
  for d in ab/*/ab-*; do
    python scripts/k8s_report.py kind --app flowsentry --out-dir $d --artifact $d/report.json --no-gate
  done
  python scripts/k8s_drain_ab.py ab --app flowsentry --out artifacts/k8s_drain_ab_flowsentry.json
  ```

  GitHub keeps those run files for 90 days (to about 2026-12-29); the summary JSON
  is committed so the numbers outlive them.
- Not done: any cloud deploy.
