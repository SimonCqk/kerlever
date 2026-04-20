# syntax=docker/dockerfile:1.6
#
# Kerlever Benchmarker — deterministic GPU measurement service.
#
# Base image rationale (design §7.1): CUDA 12.4 devel variant ships `ncu`
# and `nsys` alongside the driver headers. Ubuntu 24.04 (Noble) ships
# Python 3.12 in `universe` directly via `apt-get install python3`, so we
# can drop the `ppa:deadsnakes/ppa` dependency used on Jammy.

FROM nvidia/cuda:12.4.1-devel-ubuntu24.04

# Layer 2 — system deps (rare change). Ubuntu 24.04's `python3` / `python3-pip`
# map to 3.12; `curl` is required for the HEALTHCHECK below.
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-venv \
      ca-certificates \
      curl \
      git \
 && rm -rf /var/lib/apt/lists/*

# Layer 3 — non-root user.
RUN groupadd --gid 1000 bench \
 && useradd  --uid 1000 --gid 1000 --create-home --shell /bin/bash bench

# Layer 4 — python deps (moderate change).
WORKDIR /app
RUN python3 -m pip install --no-cache-dir --upgrade pip --break-system-packages \
 && python3 -m pip install --no-cache-dir --break-system-packages \
      "fastapi>=0.110" \
      "uvicorn[standard]>=0.27" \
      "cuda-python>=12.3" \
      "pynvml>=11.5" \
      "numpy>=1.26" \
      "pydantic>=2.5" \
      "anthropic>=0.40" \
      "pyyaml>=6.0"

# Layer 5 — source (frequent change).
COPY pyproject.toml /app/pyproject.toml
COPY kerlever/ /app/kerlever/
COPY docker/benchmarker/entrypoint.sh /app/entrypoint.sh
RUN python3 -m pip install --no-cache-dir --no-deps --break-system-packages /app \
 && chmod +x /app/entrypoint.sh \
 && chown -R bench:bench /app

# Layer 6 — runtime mount for the shared cubin + ncu-rep store.
# REQ-BENCH-031 pins cubin_uri to an absolute POSIX path on a volume shared
# with the Compiler Service; declaring VOLUME here documents that requirement
# and guarantees the mount point exists inside the image.
RUN mkdir -p /srv/kerlever/artifacts \
 && chown -R bench:bench /srv/kerlever
VOLUME ["/srv/kerlever/artifacts"]

USER bench
EXPOSE 8080

# Layer 7 — container-level readiness probe. Kubernetes probes supersede this
# in production deployments, but the baked-in HEALTHCHECK keeps plain `docker
# run` and docker-compose healthy-status signals accurate (design §7.6).
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD curl -fsS http://localhost:8080/healthz || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
