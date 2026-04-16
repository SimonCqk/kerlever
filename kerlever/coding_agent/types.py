"""Coding Agent types — internal data types for the Coding Agent module.

Spec: docs/coding-agent/spec.md §5 (Internal Types)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class CodeSeverity(StrEnum):
    """Severity level for code validation issues."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class CodeIssue:
    """A single issue found during code validation.

    Implements: REQ-CA-005
    """

    severity: CodeSeverity
    message: str
    check_name: str


@dataclass(frozen=True, slots=True)
class GPUSpec:
    """Hardware specification for a target GPU architecture.

    Implements: REQ-CA-004
    """

    arch: str
    smem_per_sm_kb: int
    max_smem_per_block_kb: int
    registers_per_sm: int
    max_registers_per_thread: int
    max_warps_per_sm: int
    max_threads_per_block: int
    hbm_bandwidth_tbps: float
    l2_cache_mb: int
    supports_cp_async: bool
    supports_tma: bool
    supports_fp8: bool
    tensor_core_types: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class PlaybookTechnique:
    """A single optimization technique in the playbook.

    Implements: REQ-CA-003
    """

    name: str
    layer: int
    applicable_when: str
    expected_gain: str
    template: str
    caveats: str


@dataclass(frozen=True, slots=True)
class PlaybookLayer:
    """A layer of optimization techniques in the playbook.

    Implements: REQ-CA-003
    """

    layer_number: int
    name: str
    techniques: list[PlaybookTechnique] = field(default_factory=list)
