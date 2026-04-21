#!/usr/bin/env bash
# Kerlever Compiler Service entrypoint.
#
# Runs the toolchain probe ONCE before uvicorn. On failure, exits non-zero
# so the container is marked unhealthy (INV-CS-012 + SC-CS-010).
#
# Spec: docs/compiler-service/spec.md §6.12
# Design: docs/compiler-service/design.md §14.2
set -euo pipefail

python3 -m kerlever.compiler_service --probe-only

exec "$@"
