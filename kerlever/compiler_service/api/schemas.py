"""HTTP edge schemas — thin aliases over the service-local types.

Spec: docs/compiler-service/spec.md §6.12
Design: docs/compiler-service/design.md §12.4
"""

from __future__ import annotations

from pydantic import BaseModel

from kerlever.compiler_service.types import PodHealth, ToolchainInfo


class HealthzResponse(BaseModel):
    """Response body for ``GET /healthz``."""

    status: str
    ok: bool
    missing: list[str] = []
    notes: list[str] = []
    toolchain: ToolchainInfo | None = None
    pod_health: PodHealth | None = None


class PodStatusResponse(BaseModel):
    """Response body for ``GET /v1/pod-status``."""

    pod_health: PodHealth
    ambiguous_failure_count: int
    toolchain: ToolchainInfo
    disk_used_pct: float
    artifact_count: int
    pinned_count: int
