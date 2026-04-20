#!/usr/bin/env bash
# Kerlever Benchmarker container entrypoint.
#
# `exec` replaces the shell so uvicorn receives SIGTERM directly from the
# container runtime. `--workers 1` is mandatory: multi-worker uvicorn
# would mean multiple independent LeaseManager instances and break the
# per-GPU semaphore guarantee (design §5.2). `--lifespan on` ensures the
# FastAPI startup hook performs NVML init / inventory capture.
set -euo pipefail

exec python3 -m uvicorn \
     kerlever.benchmarker.main:app \
     --host 0.0.0.0 \
     --port 8080 \
     --workers 1 \
     --lifespan on
