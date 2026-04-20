"""Kerlever Benchmarker — deterministic GPU measurement service.

Public surface used by tests, tooling, and future adapters. Nothing here
imports cuda-python or pynvml — callers that need GPU runtime must build
the app via :func:`create_app` and hit the HTTP surface.

Spec: docs/benchmarker/spec.md
Design: docs/benchmarker/design.md
"""

from __future__ import annotations

from typing import Any

from kerlever.benchmarker.config import BenchmarkerConfig
from kerlever.benchmarker.types import (
    AdapterIterationSemantics,
    ArtifactExecutionModel,
    BatchStatus,
    BenchmarkBatchRequest,
    BenchmarkBatchResult,
    CachePolicy,
    CandidateArtifactRef,
    CandidateResult,
    ClockPolicy,
    ClockPolicyMode,
    FaultClass,
    HygieneReport,
    IncumbentAnchor,
    IncumbentComparison,
    MeasurementEnvelope,
    MeasurementQualityStatus,
    MetricMode,
    PodHealth,
    ProfileStatus,
    ProfileUnavailableReason,
    ShapeMeasurementArtifact,
)


def create_app(*args: Any, **kwargs: Any) -> Any:
    """Lazily build the FastAPI app.

    Keeping this import lazy lets pure benchmarker modules and dry-run tooling
    import ``kerlever.benchmarker`` without requiring the service extra.
    """
    from kerlever.benchmarker.service import create_app as _create_app

    return _create_app(*args, **kwargs)


__all__ = [
    "AdapterIterationSemantics",
    "ArtifactExecutionModel",
    "BatchStatus",
    "BenchmarkBatchRequest",
    "BenchmarkBatchResult",
    "BenchmarkerConfig",
    "CachePolicy",
    "CandidateArtifactRef",
    "CandidateResult",
    "ClockPolicy",
    "ClockPolicyMode",
    "FaultClass",
    "HygieneReport",
    "IncumbentAnchor",
    "IncumbentComparison",
    "MeasurementEnvelope",
    "MeasurementQualityStatus",
    "MetricMode",
    "PodHealth",
    "ProfileStatus",
    "ProfileUnavailableReason",
    "ShapeMeasurementArtifact",
    "create_app",
]
