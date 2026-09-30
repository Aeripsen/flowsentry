"""The Kubernetes and Terraform deploy files are wired correctly. This is the
offline guard; .github/workflows/k8s.yml then deploys them to a real kind
cluster. A broken selector, probe path, uid or ConfigMap key turns CI red here
instead of failing at deploy time."""
import re
from pathlib import Path

import yaml

from flowsentry.config import Settings
from flowsentry.drain import DRAIN_FILE

REPO = Path(__file__).resolve().parents[1]


def _k8s_docs() -> dict[str, dict]:
    text = (REPO / "deploy" / "k8s" / "base" / "flowsentry.yaml").read_text()
    return {d["kind"]: d for d in yaml.safe_load_all(text) if d}


def _container() -> dict:
    return _k8s_docs()["Deployment"]["spec"]["template"]["spec"]["containers"][0]


def test_service_and_deployment_select_the_same_pods() -> None:
    docs = _k8s_docs()
    pod_labels = docs["Deployment"]["spec"]["template"]["metadata"]["labels"]
    assert docs["Deployment"]["spec"]["selector"]["matchLabels"] == pod_labels
    assert docs["Service"]["spec"]["selector"] == pod_labels
    assert docs["PodDisruptionBudget"]["spec"]["selector"]["matchLabels"] == pod_labels


def test_probes_hit_the_right_endpoints_on_the_named_port() -> None:
    c = _container()
    port = c["ports"][0]
    assert port["containerPort"] == 8000
    assert c["livenessProbe"]["httpGet"]["path"] == "/health"
    assert c["readinessProbe"]["httpGet"]["path"] == "/ready"
    assert c["livenessProbe"]["httpGet"]["port"] == port["name"]
    assert c["readinessProbe"]["httpGet"]["port"] == port["name"]
    assert _k8s_docs()["Service"]["spec"]["ports"][0]["targetPort"] == port["name"]


def test_nonroot_uid_matches_the_dockerfile() -> None:
    sec = _k8s_docs()["Deployment"]["spec"]["template"]["spec"]["securityContext"]
    assert sec["runAsNonRoot"] is True
    dockerfile = (REPO / "Dockerfile").read_text()
    assert f"--uid {sec['runAsUser']}" in dockerfile
    assert "USER app" in dockerfile


def test_rollout_never_drops_capacity_and_shutdown_is_graceful() -> None:
    dep = _k8s_docs()["Deployment"]
    rolling = dep["spec"]["strategy"]["rollingUpdate"]
    assert rolling["maxUnavailable"] == 0
    assert rolling["maxSurge"] >= 1
    pod = dep["spec"]["template"]["spec"]
    shell, flag, script = _container()["lifecycle"]["preStop"]["exec"]["command"]
    assert (shell, flag) == ("sh", "-c")
    # preStop must create the file DrainMiddleware watches, then outwait kube-proxy
    assert f"touch {DRAIN_FILE.as_posix()}" in script
    sleep_s = int(re.search(r"sleep (\d+)", script).group(1))
    assert sleep_s < pod["terminationGracePeriodSeconds"]
    # exec-form CMD: uvicorn is PID 1 and receives SIGTERM directly
    assert 'CMD ["uvicorn"' in (REPO / "Dockerfile").read_text()


def test_configmap_is_wired_and_matches_code_defaults() -> None:
    cm = _k8s_docs()["ConfigMap"]
    assert {"configMapRef": {"name": cm["metadata"]["name"]}} in _container()["envFrom"]
    serving = Settings().serving
    for key, value in cm["data"].items():
        section, field = key.removeprefix("FLOWSENTRY_").lower().split("__")
        assert section == "serving"
        assert int(value) == getattr(serving, field), key


def test_hpa_targets_the_deployment() -> None:
    docs = _k8s_docs()
    hpa = docs["HorizontalPodAutoscaler"]["spec"]
    assert hpa["scaleTargetRef"]["name"] == docs["Deployment"]["metadata"]["name"]
    assert hpa["minReplicas"] <= docs["Deployment"]["spec"]["replicas"] <= hpa["maxReplicas"]
    assert _container()["resources"]["requests"]["cpu"]
    assert docs["PodDisruptionBudget"]["spec"]["minAvailable"] < hpa["minReplicas"]


def test_terraform_settings_match_the_yaml_configmap() -> None:
    tf = (REPO / "deploy" / "terraform" / "kubernetes" / "variables.tf").read_text()
    tf_settings = dict(re.findall(r'(FLOWSENTRY_\w+)\s*=\s*"([^"]*)"', tf))
    assert tf_settings == _k8s_docs()["ConfigMap"]["data"]


def test_kind_overlay_changes_max_batch_so_ci_can_prove_the_wiring() -> None:
    text = (REPO / "deploy" / "k8s" / "overlays" / "kind" / "kustomization.yaml").read_text()
    value = yaml.safe_load(yaml.safe_load(text)["patches"][0]["patch"])[0]["value"]
    assert value != _k8s_docs()["ConfigMap"]["data"]["FLOWSENTRY_SERVING__MAX_BATCH_ROWS"]
