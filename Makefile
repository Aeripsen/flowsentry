# Convenience targets; every one is a documented single command, so Windows
# users without make can run the underlying line directly.

.PHONY: train test lint type bench splits reproduce serve

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

# the reproducibility contract: retrain and require artifacts/metrics.json to
# regenerate byte-identically (exact bytes promised under requirements.lock)
reproduce:
	python scripts/reproduce.py

serve:
	uvicorn flowsentry.service:app
