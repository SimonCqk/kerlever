"""Spec Builder LLM judge — Stage 2 semantic validation.

Evaluates five semantic dimensions of a ProblemSpec using an LLM.

Spec: docs/spec-builder/spec.md §6.3
"""

from __future__ import annotations

import json
import re

import yaml

from kerlever.llm_client import LLMClientProtocol
from kerlever.spec_builder.types import ValidationIssue
from kerlever.types import ProblemSpec

# Re-export for backward compatibility
__all__ = ["LLMClientProtocol", "run_llm_judge"]

_LLM_JUDGE_DIMENSIONS: frozenset[str] = frozenset(
    {
        "consistency",
        "specificity",
        "feasibility",
        "completeness",
        "kernel_quality",
    }
)

_VALID_SEVERITIES: frozenset[str] = frozenset({"pass", "warn", "fail"})

_SYSTEM_PROMPT = """\
You are a CUDA kernel optimization specification reviewer. You will receive \
a ProblemSpec in YAML format describing a kernel optimization target.

Evaluate the specification across exactly five dimensions and return a JSON \
array. Each element must have: "dimension", "severity", and "reason".

Dimensions:
- consistency: Do the fields form a coherent specification? (e.g., do the \
shape_cases dims match the op_semantics, does the reference kernel signature \
match the dtype and shapes, is the objective metric consistent with the \
shape_cases weights?)
- specificity: Is the specification precise enough for an optimization agent \
to act on? (e.g., are op_semantics unambiguous, is the target_metric_value \
concrete, do shape_cases cover representative workload points?)
- feasibility: Is the target_metric_value achievable given the hardware and \
operation? (Use roofline reasoning: is the target_metric_value within the \
theoretical peak for the target GPU and operation type given the shape_cases?)
- completeness: Are all fields populated with meaningful content, or are any \
fields placeholder stubs? Do shape_cases cover enough of the workload surface?
- kernel_quality: Is the reference kernel a reasonable starting point? (Does \
it implement the stated operation, is it syntactically plausible CUDA, would \
it compile?)

Severity values: "pass", "warn", "fail".

Return ONLY a JSON array with exactly 5 objects, one per dimension. Example:
[
  {"dimension": "consistency", "severity": "pass", "reason": "..."},
  {"dimension": "specificity", "severity": "pass", "reason": "..."},
  {"dimension": "feasibility", "severity": "warn", "reason": "..."},
  {"dimension": "completeness", "severity": "pass", "reason": "..."},
  {"dimension": "kernel_quality", "severity": "pass", "reason": "..."}
]

Do not include any other text outside the JSON array.\
"""


async def run_llm_judge(
    spec: ProblemSpec,
    client: LLMClientProtocol,
) -> list[ValidationIssue]:
    """Run the LLM semantic judge on a resolved ProblemSpec.

    Sends the spec to the LLM for evaluation across five semantic
    dimensions. Retries once on parse failure. Degrades to a
    parse_error issue after two failures.

    Args:
        spec: A ProblemSpec with resolved reference_kernel.
        client: An LLM client implementing LLMClientProtocol.

    Returns:
        A list of ValidationIssue objects (5 on success, or 1
        parse_error on complete failure).

    Implements: REQ-SB-004, SCN-SB-004-01, SCN-SB-004-02, SCN-SB-004-03
    Invariant: INV-SB-003 (LLM failures never crash the pipeline)
    """
    user_prompt = yaml.dump(
        spec.model_dump(),
        default_flow_style=False,
        allow_unicode=True,
    )

    last_error = ""
    for attempt in range(2):
        try:
            raw_response = await client.complete(_SYSTEM_PROMPT, user_prompt)
            issues = _parse_llm_response(raw_response)
            return issues
        except Exception as exc:  # noqa: BLE001
            last_error = f"Attempt {attempt + 1}: {exc}"

    return [
        ValidationIssue(
            dimension="parse_error",
            severity="fail",
            message=f"LLM judge failed after 2 attempts: {last_error}",
        )
    ]


def _parse_llm_response(raw: str) -> list[ValidationIssue]:
    """Parse the LLM response into ValidationIssue objects.

    Strips markdown code fences if present, then parses JSON.

    Raises:
        ValueError: If the response cannot be parsed or validated.
    """
    cleaned = _strip_markdown_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse LLM response as JSON: {exc}") from exc

    if not isinstance(data, list) or len(data) != 5:
        raise ValueError(
            f"Expected JSON array of 5 items, got {type(data).__name__} "
            f"with {len(data) if isinstance(data, list) else 'N/A'} items"
        )

    seen_dimensions: set[str] = set()
    issues: list[ValidationIssue] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError(f"Expected JSON object, got {type(item).__name__}")

        dimension = item.get("dimension")
        severity = item.get("severity")
        reason = item.get("reason")

        if dimension not in _LLM_JUDGE_DIMENSIONS:
            raise ValueError(f"Unknown dimension: {dimension!r}")
        if severity not in _VALID_SEVERITIES:
            raise ValueError(f"Invalid severity: {severity!r}")
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"Missing or empty reason for dimension {dimension!r}")

        if dimension in seen_dimensions:
            raise ValueError(f"Duplicate dimension: {dimension!r}")
        seen_dimensions.add(dimension)

        # dimension and severity are validated above to be valid literals
        issue = ValidationIssue.model_validate(
            {
                "dimension": dimension,
                "severity": severity,
                "message": reason,
            }
        )
        issues.append(issue)

    if seen_dimensions != _LLM_JUDGE_DIMENSIONS:
        missing = _LLM_JUDGE_DIMENSIONS - seen_dimensions
        raise ValueError(f"Missing dimensions: {missing}")

    return issues


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from a string."""
    stripped = text.strip()
    pattern = re.compile(r"^```(?:json)?\s*\n(.*)\n\s*```$", re.DOTALL)
    match = pattern.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped
