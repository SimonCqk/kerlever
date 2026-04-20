"""Kerlever Benchmarker — deterministic GPU measurement service.

Public surface used by tests, tooling, and future adapters. Nothing here
imports cuda-python or pynvml — callers that need GPU runtime must build
the app via :func:`create_app` and hit the HTTP surface.

Spec: docs/benchmarker/spec.md
Design: docs/benchmarker/design.md
"""

from __future__ import annotations

from kerlever.benchmarker.config import BenchmarkerConfig
from kerlever.benchmarker.service import create_app
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
