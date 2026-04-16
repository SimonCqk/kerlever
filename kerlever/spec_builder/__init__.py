"""Spec Builder — validation and interactive collection for ProblemSpec.

Public API: ``validate_spec()`` and ``interactive_collect()``.

Spec: docs/spec-builder/spec.md
"""

from __future__ import annotations

from kerlever.spec_builder.deterministic import run_deterministic_checks
from kerlever.spec_builder.interactive import interactive_collect
from kerlever.spec_builder.llm_judge import LLMClientProtocol, run_llm_judge
from kerlever.spec_builder.resolver import resolve_reference_kernel
from kerlever.spec_builder.types import ValidationIssue, ValidationResult
from kerlever.types import ProblemSpec

__all__ = [
    "interactive_collect",
    "validate_spec",
]


async def validate_spec(
    spec: ProblemSpec,
    *,
    llm_client: LLMClientProtocol | None = None,
) -> ValidationResult:
    """Run the full validation pipeline on a ProblemSpec.

    Stage 1 (deterministic) always runs. Stage 2 (LLM judge) runs only
    if an ``llm_client`` is provided.

    The pipeline:
    1. Resolve the reference kernel (inline/file/URL).
    2. Run deterministic checks on the resolved spec.
    3. If llm_client is provided, run the LLM judge.
    4. Merge all issues and compute is_valid.

    Args:
        spec: The ProblemSpec to validate.
        llm_client: Optional LLM client for Stage 2 semantic validation.
            Pass None to skip Stage 2 (equivalent to --no-llm).

    Returns:
        A ValidationResult with is_valid and a list of issues.

    Implements: REQ-SB-001, REQ-SB-003
    Invariant: INV-SB-001 (reference kernel resolved before validation)
    """
    issues: list[ValidationIssue] = []

    # Step 1: Resolve reference kernel (INV-SB-001)
    try:
        resolved_kernel = await resolve_reference_kernel(spec.reference_kernel)
        spec = spec.model_copy(update={"reference_kernel": resolved_kernel})
    except (FileNotFoundError, ValueError) as exc:
        issues.append(
            ValidationIssue(
                dimension="reference_kernel",
                severity="fail",
                message=str(exc),
            )
        )
        return ValidationResult(issues=issues)

    # Step 2: Deterministic checks (INV-SB-002)
    issues.extend(run_deterministic_checks(spec))

    # Step 3: LLM judge (if enabled)
    if llm_client is not None:
        llm_issues = await run_llm_judge(spec, llm_client)
        issues.extend(llm_issues)

    return ValidationResult(issues=issues)
