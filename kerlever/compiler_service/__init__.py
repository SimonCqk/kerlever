"""Compiler Service — deterministic GPU gate for CUDA kernel candidates.

The Compiler Service decides whether a generated CUDA kernel is executable
and correct enough to be measured. It sits between the Coding Agent (which
emits candidate source) and the Benchmarker (which measures performance).

Public API: ``CompilerService``, ``CompileRequest``, ``CompileResult``,
``create_app``.

Spec: docs/compiler-service/spec.md
Design: docs/compiler-service/design.md
"""

from __future__ import annotations

from typing import Any

from kerlever.compiler_service.service import CompilerService, CompilerServiceDeps
from kerlever.compiler_service.types import (
    CandidateFaultKind,
    CandidateRole,
    CompileRequest,
    CompileResult,
    CompileResultStatus,
    FaultClass,
    IdempotencyState,
    KernelExecutionSpec,
    MetadataMode,
    PodHealth,
    RequestLimits,
    RunEnvelope,
    SanitizerOutcome,
    SanitizerStatus,
    SanitizerTool,
    StaticAnalysisExt,
    ToolchainInfo,
)

__all__ = [
    "CandidateFaultKind",
    "CandidateRole",
    "CompileRequest",
    "CompileResult",
    "CompileResultStatus",
    "CompilerService",
    "CompilerServiceDeps",
    "FaultClass",
    "IdempotencyState",
    "KernelExecutionSpec",
    "MetadataMode",
    "PodHealth",
    "RequestLimits",
    "RunEnvelope",
    "SanitizerOutcome",
    "SanitizerStatus",
    "SanitizerTool",
    "StaticAnalysisExt",
    "ToolchainInfo",
    "create_app",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve optional service-only symbols.

    ``create_app`` lives in :mod:`kerlever.compiler_service.api.app` and
    depends on ``fastapi``, which is only installed via the ``[service]``
    optional extra. Importing it eagerly would break CPU-only callers that
    only want the core types (e.g. ``CompileRequest``). We therefore expose
    it through a module-level ``__getattr__`` so the import happens on first
    attribute access, after the caller has already opted in to the service
    extra.
    """
    if name == "create_app":
        try:
            from kerlever.compiler_service.api.app import create_app
        except ImportError as exc:  # pragma: no cover — optional extra
            raise ImportError(
                "create_app requires the optional service extra; "
                "install `kerlever[service]` to use it"
            ) from exc
        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
