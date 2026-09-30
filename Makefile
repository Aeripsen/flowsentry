# Convenience targets; every one is a documented single command, so Windows
# users without make can run the underlying line directly.

.PHONY: train test lint type bench splits hierarchy families calibration gbdt shap zero-day business reproduce serve

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

# the reproducibility contract: retrain and require artifacts/metrics.json to
# regenerate byte-identically (exact bytes promised under requirements.lock)
reproduce:
	python scripts/reproduce.py

serve:
	uvicorn flowsentry.service:app
