# FlowSentry serving image. Trains inside the build from the committed BCCC sample,
# so every image is self-contained and /predict works on a clean clone.
FROM python:3.12-slim

WORKDIR /app

# Install deps + the package editable, so the package stays under /app/src and its
# path-relative data/ and artifacts/ dirs resolve to /app (not site-packages).
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[dashboard]"

# App code, dashboard, and the committed data sample (needed to train at build time).
COPY dashboard ./dashboard
COPY data ./data

# Train the model into artifacts/ so the image ships with a working model.
RUN python -m flowsentry.train

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Liveness: hit /health with stdlib urllib (curl is not in the slim image).
# /health returns 503 when the model is missing, so an unhealthy container is
# reported unhealthy rather than falsely green.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "flowsentry.service:app", "--host", "0.0.0.0", "--port", "8000"]
