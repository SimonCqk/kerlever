# Kerlever Compiler Service — CUDA devel image.
#
# Spec: docs/compiler-service/spec.md §6.12, REQ-CS-010, REQ-CS-013
# Design: docs/compiler-service/design.md §14

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# System deps: Python 3.12 + build essentials for cuda-python wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         python3.12 \
         python3.12-venv \
         python3-pip \
         ca-certificates \
         git \
    && rm -rf /var/lib/apt/lists/*

# compute-sanitizer ships at /usr/local/cuda/bin/compute-sanitizer.
ENV PATH="/usr/local/cuda/bin:${PATH}"

# Install the package with the [service] optional extra.
COPY . /opt/kerlever
WORKDIR /opt/kerlever
RUN python3.12 -m pip install --no-cache-dir --upgrade pip \
    && python3.12 -m pip install --no-cache-dir ".[service]"

# Artifact root (matches ServiceConfig default).
ENV KERLEVER_ARTIFACT_ROOT=/var/lib/kerlever/artifacts
RUN mkdir -p "${KERLEVER_ARTIFACT_ROOT}" \
    && chmod 0750 "${KERLEVER_ARTIFACT_ROOT}"

EXPOSE 8080
ENTRYPOINT ["/opt/kerlever/docker/compiler-service/entrypoint.sh"]
CMD ["uvicorn", "kerlever.compiler_service.api.app:create_app", \
     "--factory", "--host", "0.0.0.0", "--port", "8080"]
