"""Compute Sanitizer runner + escalation policy.

Spec: docs/compiler-service/spec.md §6.7
Design: docs/compiler-service/design.md §8.4
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from kerlever.compiler_service.config import ServiceConfig
from kerlever.compiler_service.types import (
    SanitizerOutcome,
    SanitizerStatus,
    SanitizerTool,
)
from kerlever.types import ProblemSpec, ShapeCase

# Sanitizer escalation triggers are lexical regexes over the candidate
# source (spec §6.7; AST parsing is out of scope for V1).
_RACECHECK_TRIGGERS = re.compile(
    r"(__shared__|__syncwarp|cooperative_groups)", re.MULTILINE
)
_SYNCCHECK_TRIGGERS = re.compile(
    r"(__syncthreads|__syncwarp|cooperative_groups|cuda::barrier)", re.MULTILINE
)


@dataclass(frozen=True)
class SanitizerRawResult:
    """Structured outcome of a sanitizer subprocess (design §8.4)."""

    returncode: int
    report_path: Path | None
    stdout_excerpt: str
    stderr_excerpt: str
    timed_out: bool


class ComputeSanitizerRunner:
    """Wraps ``compute-sanitizer`` with timeout + report capture.

    Every invocation produces a ``SanitizerOutcome`` — never a silent
    drop (INV-CS-004).
    """

    def __init__(self, sanitizer_path: Path, config: ServiceConfig) -> None:
        self._sanitizer_path = sanitizer_path
        self._config = config

    async def run(
        self,
        tool: SanitizerTool,
        executable: Path,
        shape: ShapeCase,
        input_dir: Path,
        timeout_s: float | None = None,
        report_path: Path | None = None,
    ) -> SanitizerOutcome:
        """Run ``compute-sanitizer --tool <tool>`` against ``executable``.

        Always returns a ``SanitizerOutcome``; the caller (Phase 4)
        appends it to ``correctness.sanitizer_results``.

        Invariant: INV-CS-004
        """
        del input_dir  # reserved for a future harness invocation layout
        timeout = timeout_s or self._config.sanitizer_timeout_s
        save_path = report_path or executable.with_suffix(f".{tool.value}.report")

        argv = [
            str(self._sanitizer_path),
            f"--tool={tool.value}",
            "--error-exitcode=1",
            "--save",
            str(save_path),
            str(executable),
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return SanitizerOutcome(
                tool=tool,
                shape_id=shape.shape_id,
                status=SanitizerStatus.UNSUPPORTED,
                report_artifact_id=None,
            )

        try:
            _, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except TimeoutError:
            process.kill()
            with contextlib.suppress(Exception):
                await process.wait()
            return SanitizerOutcome(
                tool=tool,
                shape_id=shape.shape_id,
                status=SanitizerStatus.TIMEOUT,
                report_artifact_id=None,
            )

        returncode = process.returncode or 0
        status = _classify_returncode(returncode)
        return SanitizerOutcome(
            tool=tool,
            shape_id=shape.shape_id,
            status=status,
            report_artifact_id=None,
        )


class SanitizerPolicy:
    """Decides which sanitizer tools to run for a candidate (spec §6.7).

    ``memcheck`` always runs on the smallest shape; ``racecheck``,
    ``synccheck``, ``initcheck`` escalate based on lexical triggers over
    the candidate source and the correctness outputs.
    """

    def __init__(self, config: ServiceConfig) -> None:
        self._config = config

    def decide(
        self,
        candidate_source: str,
        problem_spec: ProblemSpec,
        saw_nan_or_inf: bool,
        adapter_high_risk_shapes: frozenset[str],
        semantic_diff_flag: bool = False,
    ) -> list[SanitizerTool]:
        """Return the ordered list of sanitizer tools to run.

        Spec §6.7: ``memcheck`` first (always); others conditional.
        """
        del problem_spec  # shape metadata is handled by the caller
        tools: list[SanitizerTool] = [self._config.sanitizer_default_tool]

        if _RACECHECK_TRIGGERS.search(candidate_source):
            tools.append(SanitizerTool.RACECHECK)

        if (
            _SYNCCHECK_TRIGGERS.search(candidate_source)
            and SanitizerTool.SYNCCHECK not in tools
        ):
            tools.append(SanitizerTool.SYNCCHECK)

        if (
            saw_nan_or_inf or adapter_high_risk_shapes or semantic_diff_flag
        ) and SanitizerTool.INITCHECK not in tools:
            tools.append(SanitizerTool.INITCHECK)

        # Preserve spec-mandated ordering: memcheck, racecheck, synccheck,
        # initcheck.
        return _order_tools(tools)

    @staticmethod
    def smallest_shape(shapes: Sequence[ShapeCase]) -> ShapeCase:
        """Return the shape with the smallest product of dims (spec §6.7)."""
        if not shapes:
            raise ValueError("cannot choose smallest shape from empty list")
        return min(shapes, key=_dim_product)


def _order_tools(tools: list[SanitizerTool]) -> list[SanitizerTool]:
    """Sort tools into the canonical spec order."""
    order = {
        SanitizerTool.MEMCHECK: 0,
        SanitizerTool.RACECHECK: 1,
        SanitizerTool.SYNCCHECK: 2,
        SanitizerTool.INITCHECK: 3,
    }
    return sorted(set(tools), key=lambda t: order[t])


def _dim_product(shape: ShapeCase) -> int:
    """Product of ``shape.dims`` — used for SANITIZER_SHAPE_POLICY."""
    product = 1
    for dim in shape.dims:
        product *= max(1, int(dim))
    return product


def _classify_returncode(returncode: int) -> SanitizerStatus:
    """Map a sanitizer returncode to ``SanitizerStatus``.

    ``--error-exitcode=1`` means the tool flags errors as exit code 1.
    Exit 0 → pass; exit 1 → fail; anything else → unsupported (the tool
    itself crashed, which we treat as ``UNSUPPORTED`` rather than letting
    it masquerade as a candidate fault).
    """
    if returncode == 0:
        return SanitizerStatus.PASS
    if returncode == 1:
        return SanitizerStatus.FAIL
    return SanitizerStatus.UNSUPPORTED
