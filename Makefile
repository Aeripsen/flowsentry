# Convenience targets; every one is a documented single command, so Windows
# users without make can run the underlying line directly.

.PHONY: train test lint type bench splits hierarchy families calibration gbdt shap zero-day business verify-derived demo-data demo-verify site-check reproduce serve track drift-report mlflow-ui k8s-e2e tf-kind

train:
	python -m flowsentry.train

test:
	pytest -q

lint:
	ruff check src tests scripts dashboard

type:
	mypy

bench:
	python -m flowsentry.bench

# grouped vs stratified head to head; sources the split claims in ADR 002
splits:
	python scripts/split_comparison.py

# what the two-stage hierarchy buys vs single joint models; sources ADR 001
hierarchy:
	python scripts/hierarchy_benchmark.py

# per-family precision/recall at full coverage and under the reject knob
families:
	python scripts/per_family_report.py

# reliability curve, ECE/MCE and Brier for the shipped model's confidence
calibration:
	python scripts/calibration_report.py

# boosted trees vs the shipped forest, printed grid + calibration; needs [gbdt]
gbdt:
	python scripts/gbdt_comparison.py

# SHAP attribution for the gbdt comparison winner; needs [gbdt], run gbdt first
shap:
	python scripts/shap_attribution.py

# leave-one-family-out: what the model does with an attack family it never saw,
# with a shuffled-label control. The open-set case, measured
zero-day:
	python scripts/zero_day_lofo.py

# the reject knob as verdicts per 10,000 flows (answered, to an analyst, wrong)
# and the zero-day silent-miss rate, derived from the committed artifacts above;
# trains nothing, prices nothing
business:
	python scripts/business_case.py

# per-flow export the live demo page runs on; refuses to write unless the rows
# rebuild the committed curve, binary PR-AUC and per-family confusion
demo-data:
	python scripts/demo_data.py

# check that a fresh export equals the committed one byte for byte
# (trains first if there is no local model); CI's `demo` job
demo-verify:
	python scripts/demo_data.py --verify

# load the demo page in headless Chromium and check what it displays against
# the committed artifacts; needs playwright + `python -m playwright install chromium`
site-check:
	python scripts/check_site.py

# the reproducibility contract: retrain and require artifacts/metrics.json to
# regenerate byte-identically (exact bytes promised under requirements.lock)
reproduce:
	python scripts/reproduce.py

# per_family.json and zero_day_lofo.json regenerate unchanged (retrains; the
# environment stamp is the only key allowed to differ). CI runs this.
verify-derived:
	python scripts/verify_derived.py

serve:
	uvicorn flowsentry.service:app

# MLflow: the model comparison (two-stage forest, single joint forest, tuned
# XGBoost, tuned LightGBM) as one run per arm, cross-checked against the committed
# artifacts; needs [gbdt,mlops]. `make train` also logs a run once mlflow is installed
track:
	python scripts/track_arms.py

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

# Evidently: data drift over the 132 features + classification quality, training
# connections (out-of-fold) vs held-out connections -> reports/*.html; needs [mlops]
drift-report:
	python scripts/evidently_report.py

# Deploy to a throwaway kind cluster, smoke + load test, rolling restart under
# load (needs docker, kind, kubectl). Same script the k8s CI workflow runs.
k8s-e2e:
	bash scripts/k8s_e2e.sh

# terraform apply deploy/terraform/kubernetes to kind, re-plan, parity, destroy.
tf-kind:
	bash scripts/tf_kind_e2e.sh
