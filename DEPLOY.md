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
| Kubernetes (kind) | Deployed, smoke-tested, load-tested and rolled under load in CI on every deploy change | [`.github/workflows/k8s.yml`](.github/workflows/k8s.yml), [`artifacts/k8s_kind_flowsentry.json`](artifacts/k8s_kind_flowsentry.json) |
| Terraform, `hashicorp/kubernetes` | Applied to kind, re-planned, diffed against the YAML, destroyed, in CI | [`artifacts/terraform_kind_flowsentry.json`](artifacts/terraform_kind_flowsentry.json) |
| Any cloud | **Not deployed** | no cloud account is used |
| docker compose | Runs locally | section 3 |

Everything in CI runs on a free GitHub-hosted runner (4 vCPU, 16 GB).

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
```

That script is exactly what CI runs:

1. `docker build -t flowsentry:ci .` (trains the model), `kind create cluster`,
   `kind load docker-image`
2. install metrics-server (the HPA needs it), `kubectl apply -k deploy/k8s/overlays/kind`,
   `kubectl rollout status` (returns once every pod passed `/ready`)
3. through the Service: `/health`, `/ready`, `/predict`, and `/openapi.json`, whose
   `/predict/batch` `maxItems` must equal the ConfigMap value; record PID 1 and the uid
4. steady load: 16 k6 virtual users for 90 s, closed loop, while logging the HPA
5. rolling restart under load: `kubectl rollout restart` 15 s into a 120 s k6 run
6. [`scripts/k8s_report.py`](scripts/k8s_report.py) writes the artifact and **fails
   the job on any failed request**, an unreplaced pod, a rollout that outlasted the
   load, a ConfigMap value that did not reach the process, or a shell as PID 1

To deploy the base to a real cluster instead, push the image somewhere the cluster
can pull, set `image:` and run `kubectl apply -k deploy/k8s/base`.

### Captured output

From CI run [36771699879](https://github.com/Aeripsen/flowsentry/actions/runs/36771699879)
on master (commit `849874a`), kind v0.33.0, Kubernetes v1.37.0. The full JSON is
[`artifacts/k8s_kind_flowsentry.json`](artifacts/k8s_kind_flowsentry.json).

```text
rollout: 2 pods ready in 6 s
PID 1:   /usr/local/bin/python3.12 /usr/local/bin/uvicorn flowsentry.service:app ...
runs as: uid=10001(app)
ConfigMap FLOWSENTRY_SERVING__MAX_BATCH_ROWS=2000, served /predict/batch maxItems=2000
/predict: {"label": "unknown", "confidence": 0.5251, "escalated_to_stage2": true, "abstained": true}

steady (16 VUs, 90 s):  12,858 requests, 142.7 req/s, p50 100.5 ms, p99 319.8 ms, 0 failed
rolling restart (120 s): 43,980 requests, 366.3 req/s, p50 36.2 ms, p99 125.3 ms, 0 failed
  restart began 15 s into the load, took 37 s, finished before the load ended,
  all 5 pods replaced
HPA: 2 -> 5 replicas, CPU peaked at 377% of the 250m request
```

The `/predict` answer is for the example flow in the API docs with
`reject_threshold: 0.9`; its confidence is below the threshold, so it abstains.
That is the reject option working, not an error.

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
a pod into drain mode. Checked on real sockets that uvicorn closes the TCP
connection after such a response with both of its HTTP parsers (h11, which this
image uses, and httptools).

After the fix, three runs all passed with 0 failed requests: 40,139 and 37,164
([run 36770459936](https://github.com/Aeripsen/flowsentry/actions/runs/36770459936),
attempts 1 and 2) and 43,980 (the master run above), 121,283 in total. On its own
that is weak evidence for a race this rare: if the fix did nothing and the rate
stayed at 1 per 40,070, zero failures in 121,283 would still happen about 5% of the
time. The stronger evidence is the LedgerSentry side of the same fix (0 in 254,944
after 4 in 114,308) and the socket-level check.

**Keep-alive also defeats the autoscaler's new pods.** In all four runs the HPA
reached 5 replicas 36 to 66 s after the load started, but at the end of the steady
phase `kubectl top` showed only 2 of the 5 pods working (about 1000m each, the CPU
limit) and 3 idle at 2m, every time. The 16 k6 connections were opened before
the scale-up and stayed on the pods that existed then, because kube-proxy balances
connections, not requests. The rolling restart forced every client to reconnect,
which spread them over all 5 pods, and throughput rose 2.6x (142.7 to 366.3 req/s)
in the master run. A real deployment would fix this with an L7 proxy or service
mesh that balances per request. Not done here.

**Per-pod HTTP throughput is far below the scoring benchmark.** With 2 busy pods
each capped at 1 CPU, the steady phase works out to about 53 to 85 req/s per busy
pod across the four runs, while
`python -m flowsentry.bench` measures about 415 flows/s for sequential single-flow
scoring on the 12-core dev machine. The gap between those two numbers has not been
profiled. The candidates (the CPU limit, a slower runner CPU, JSON parsing and
validation of the feature dict, one structured log line per request, the sync
endpoint's thread pool) are untested, so none is claimed.

**Memory** was about 265 to 271 Mi per pod under load, against a 384Mi request and
a 1Gi limit.

### What these numbers are not

- One single-node kind cluster on one 4-vCPU runner, with the k6 load generator in
  the same cluster, competing for the same CPUs as the pods.
- Synthetic load: one fixed flow in a closed loop. Not production traffic, not a
  live network tap.
- Throughput varies between runners. Across the four runs above, steady-phase
  throughput was 105.9, 107.8, 169.5 and 142.7 req/s. The CI gate is therefore on
  failed requests, never on a throughput number.

---

## 2. Terraform (kubernetes provider, applied in CI)

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
(the live Terraform objects against a server-side dry run of `deploy/k8s/base`, so
the two definitions cannot drift), then `terraform destroy`.

From the same master run, Terraform 1.16.4, provider 3.2.1
([`artifacts/terraform_kind_flowsentry.json`](artifacts/terraform_kind_flowsentry.json)):

```text
Apply complete! Resources: 6 added, 0 changed, 0 destroyed.      (8 s)
second plan: No changes. Your infrastructure matches the configuration.
parity: 5 objects compared, 0 mismatches
Destroy complete! Resources: 6 destroyed.
```

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

---

## Validation status

- CI, every push that touches a deploy file: the kind deployment, load test and
  rolling restart (section 1), the Terraform apply/re-plan/parity/destroy
  (section 2), and `terraform fmt -check` plus `validate`.
- CI, every push: `tests/test_deploy_manifests.py` pins the contracts offline:
  selectors, probe paths, named ports, the non-root uid against the Dockerfile,
  `maxUnavailable: 0`, the preStop drain path against `drain.py`, the ConfigMap
  keys against the code defaults, the HPA and PDB targets, and the Terraform
  settings against the YAML ConfigMap.
- Locally, before the first push: `kubeconform -strict` against the Kubernetes 1.33
  schemas on the base and the kind overlay.
- Not done: any cloud deploy.
