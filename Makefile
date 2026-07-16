# Convenience targets; every one is a documented single command, so Windows
# users without make can run the underlying line directly.

.PHONY: train test lint type bench reproduce serve

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

# the reproducibility contract: retrain and require artifacts/metrics.json to
# regenerate byte-identically (exact bytes promised under requirements.lock)
reproduce:
	python scripts/reproduce.py

serve:
	uvicorn flowsentry.service:app
