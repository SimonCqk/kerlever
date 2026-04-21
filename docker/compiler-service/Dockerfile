# Kerlever Compiler Service — CUDA devel image.
#
# Spec: docs/compiler-service/spec.md §6.12, REQ-CS-010, REQ-CS-013
# Design: docs/compiler-service/design.md §14

ARG CUDA_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

# System deps: Ubuntu 24.04 ships Python 3.12 as python3.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         python3 \
         python3-venv \
         python3-pip \
         ca-certificates \
         git \
    && rm -rf /var/lib/apt/lists/*

# compute-sanitizer ships at /usr/local/cuda/bin/compute-sanitizer.
ENV PATH="/usr/local/cuda/bin:${PATH}"

# Install the package with the [service] optional extra.
COPY . /opt/kerlever
WORKDIR /opt/kerlever
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir ".[service]"

# Artifact root (matches ServiceConfig default).
ENV KERLEVER_ARTIFACT_ROOT=/var/lib/kerlever/artifacts
RUN mkdir -p "${KERLEVER_ARTIFACT_ROOT}" \
    && chmod 0750 "${KERLEVER_ARTIFACT_ROOT}"

EXPOSE 8080
ENTRYPOINT ["/opt/kerlever/docker/compiler-service/entrypoint.sh"]
CMD ["uvicorn", "kerlever.compiler_service.api.app:create_app", \
     "--factory", "--host", "0.0.0.0", "--port", "8080"]
