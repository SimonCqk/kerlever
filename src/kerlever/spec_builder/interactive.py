"""Spec Builder interactive mode — conversational field collection.

Collects ProblemSpec fields from the user via LLM-assisted conversation.

Spec: docs/spec-builder/spec.md §6.5
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import yaml

from kerlever.spec_builder.llm_judge import LLMClientProtocol
from kerlever.spec_builder.types import ValidationResult
from kerlever.types import ProblemSpec

_REQUIRED_FIELDS: list[str] = [
    "op_name",
    "op_semantics",
    "shapes",
    "dtype",
    "target_gpu",
    "baseline_perf_us",
    "target_perf_us",
    "tolerance",
    "max_rounds",
    "reference_kernel",
]

_EXTRACTION_SYSTEM_PROMPT = """\
You are helping a user define a CUDA kernel optimization problem specification.

The ProblemSpec has these fields:
- op_name (str): Short name for the operation (e.g., "matmul", "softmax")
- op_semantics (str): Mathematical description (e.g., "C[M,N] = A[M,K] @ B[K,N]")
- shapes (list[list[int]]): Input tensor dimensions (e.g., [[4096, 4096, 4096]])
- dtype (str): Data type (float16, float32, float64, bfloat16, int8, etc.)
- target_gpu (str): GPU architecture (A100, H100, etc.)
- baseline_perf_us (float): Baseline kernel latency in microseconds
- target_perf_us (float): Target kernel latency in microseconds (must be <= baseline)
- tolerance (float): Acceptable numerical error tolerance (0 < tolerance < 1)
- max_rounds (int): Maximum optimization rounds (>= 1)
- reference_kernel (str): CUDA source code with __global__ or __device__ function

Extract any ProblemSpec fields you can identify from the user's message.
Return a JSON object with two keys:
- "extracted": an object mapping field names to values for any fields found
- "follow_up": a string with a natural language question asking for missing fields

Already collected fields: {collected}

Return ONLY valid JSON. Do not include markdown fences or other text.\
"""

_WELCOME_MESSAGE = """Welcome to the Kerlever Spec Builder interactive mode.

I'll help you define a CUDA kernel optimization problem specification.
Tell me about the kernel you want to optimize — what operation it performs,
the data types, matrix/tensor dimensions, target GPU, and performance goals.

You can describe everything at once or provide details one at a time.
"""


async def interactive_collect(
    llm_client: LLMClientProtocol,
    *,
    no_llm: bool = False,
) -> None:
    """Run the interactive ProblemSpec collection loop.

    Collects fields from the user via conversational LLM extraction,
    validates the assembled spec, and outputs YAML on success.

    Args:
        llm_client: An LLM client for field extraction.
        no_llm: If True, skip Stage 2 (LLM judge) during validation.

    Implements: REQ-SB-005, SCN-SB-005-01, SCN-SB-005-02
    Invariant: INV-SB-004 (always validates before YAML output)
    """
    # Import here to avoid circular imports at module level
    from kerlever.spec_builder import validate_spec

    collected: dict[str, Any] = {}
    print(_WELCOME_MESSAGE)

    while True:
        try:
            user_input = await asyncio.to_thread(input, "> ")
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            sys.exit(1)

        if not user_input.strip():
            continue

        # Send to LLM for field extraction
        system_prompt = _EXTRACTION_SYSTEM_PROMPT.format(
            collected=json.dumps(collected),
        )
        try:
            raw_response = await llm_client.complete(system_prompt, user_input)
            extracted_data = _parse_extraction_response(raw_response)
        except Exception as exc:  # noqa: BLE001
            print(f"Error processing input: {exc}")
            print("Please try again.")
            continue

        # Accumulate extracted fields
        new_fields = extracted_data.get("extracted", {})
        if isinstance(new_fields, dict):
            collected.update(new_fields)

        # Show follow-up
        follow_up = extracted_data.get("follow_up", "")
        if follow_up:
            print(f"\n{follow_up}\n")

        # Check if all required fields are present
        missing = [f for f in _REQUIRED_FIELDS if f not in collected]
        if missing:
            continue

        # All fields collected — attempt to build and validate
        try:
            spec = ProblemSpec.model_validate(collected)
        except Exception as exc:  # noqa: BLE001
            print(f"\nError assembling spec: {exc}")
            print("Please correct the fields above.\n")
            continue

        # Run full validation pipeline (INV-SB-004)
        result: ValidationResult = await validate_spec(
            spec,
            llm_client=None if no_llm else llm_client,
        )

        if result.is_valid:
            print("\nValidation passed! Here is your ProblemSpec:\n")
            print("---")
            print(
                yaml.dump(
                    spec.model_dump(),
                    default_flow_style=False,
                    allow_unicode=True,
                )
            )
            sys.exit(0)
        else:
            print("\nValidation failed with the following issues:")
            for issue in result.issues:
                if issue.severity == "fail":
                    print(f"  FAIL [{issue.dimension}]: {issue.message}")
                elif issue.severity == "warn":
                    print(f"  WARN [{issue.dimension}]: {issue.message}")
            print("\nWould you like to correct any fields? (yes/no)")
            try:
                answer = await asyncio.to_thread(input, "> ")
            except (EOFError, KeyboardInterrupt):
                print("\nSession ended.")
                sys.exit(1)
            if answer.strip().lower() not in ("yes", "y"):
                sys.exit(1)
            print("\nPlease provide the corrected information.\n")


def _parse_extraction_response(raw: str) -> dict[str, Any]:
    """Parse the LLM extraction response.

    Raises:
        ValueError: If the response is not valid JSON.
    """
    stripped = raw.strip()
    # Strip markdown fences if present
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        # Remove first and last lines (fences)
        inner_lines = []
        in_fence = False
        for line in lines:
            if line.strip().startswith("```") and not in_fence:
                in_fence = True
                continue
            if line.strip() == "```" and in_fence:
                break
            if in_fence:
                inner_lines.append(line)
        stripped = "\n".join(inner_lines)

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse extraction response: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")

    result: dict[str, Any] = data
    return result
