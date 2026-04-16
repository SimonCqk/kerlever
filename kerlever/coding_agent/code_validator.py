"""Coding Agent code validator — 7 regex-level CUDA code checks.

Performs structural validation of generated CUDA code without compilation.
Each check produces zero or one CodeIssue. All checks always run (no
short-circuit) so the full issue list is available for retry feedback.

Spec: docs/coding-agent/spec.md §6.4
"""

from __future__ import annotations

import re

from kerlever.coding_agent.types import CodeIssue, CodeSeverity

# Dtype mapping from ProblemSpec dtype to expected CUDA pointer types
DTYPE_MAP: dict[str, list[str]] = {
    "float16": ["half"],
    "float32": ["float"],
    "float64": ["double"],
    "bfloat16": ["nv_bfloat16"],
    "int8": ["int8_t", "char"],
    "int32": ["int"],
}

# Host-only API patterns
_HOST_APIS = [
    (re.compile(r"\bmalloc\s*\("), "malloc"),
    (re.compile(r"\bfree\s*\("), "free"),
    (re.compile(r"\bprintf\s*\("), "printf"),
    (re.compile(r"\bstd::"), "std::"),
    (re.compile(r"\bcudaMalloc\b"), "cudaMalloc"),
    (re.compile(r"\bcudaMemcpy\b"), "cudaMemcpy"),
    (re.compile(r"\bcudaFree\b"), "cudaFree"),
]

# Pattern to find __global__ function declarations
_GLOBAL_FUNC_PATTERN = re.compile(r"__global__\s+(?:void|[\w:]+)\s+\w+\s*\(")

# Pattern for __launch_bounds__
_LAUNCH_BOUNDS_PATTERN = re.compile(r"__launch_bounds__\s*\(")

# Pattern to extract the parameter list from a __global__ function
_GLOBAL_PARAMS_PATTERN = re.compile(r"__global__\s+(?:void|[\w:]+)\s+\w+\s*\(([^)]*)\)")


def _check_global_function(code: str) -> CodeIssue | None:
    """Check 1: __global__ function exists (severity: error).

    Implements: REQ-CA-005, SCN-CA-005-01
    """
    if _GLOBAL_FUNC_PATTERN.search(code):
        return None
    return CodeIssue(
        severity=CodeSeverity.ERROR,
        message="No __global__ function found in generated code.",
        check_name="global_function",
    )


def _check_launch_bounds(code: str) -> CodeIssue | None:
    """Check 2: __launch_bounds__ present (severity: warning).

    Implements: REQ-CA-005, SCN-CA-005-05
    """
    if _LAUNCH_BOUNDS_PATTERN.search(code):
        return None
    return CodeIssue(
        severity=CodeSeverity.WARNING,
        message=(
            "No __launch_bounds__ annotation found. This may cause "
            "suboptimal register allocation."
        ),
        check_name="launch_bounds",
    )


def _check_restrict_qualifiers(code: str) -> CodeIssue | None:
    """Check 3: __restrict__ on pointer parameters (severity: warning).

    Finds pointer parameters in __global__ function signatures and checks
    each has __restrict__.

    Implements: REQ-CA-005, SCN-CA-005-06
    """
    match = _GLOBAL_PARAMS_PATTERN.search(code)
    if not match:
        # If no __global__ func found, check 1 will catch it
        return None

    params_str = match.group(1)
    # Split by comma to get individual parameters
    params = [p.strip() for p in params_str.split(",") if p.strip()]

    for param in params:
        # Check if this is a pointer parameter (contains *)
        if "*" in param and "__restrict__" not in param:
            return CodeIssue(
                severity=CodeSeverity.WARNING,
                message="Pointer parameter(s) missing __restrict__ qualifier.",
                check_name="restrict_qualifier",
            )

    return None


def _check_bracket_balance(code: str) -> CodeIssue | None:
    """Check 4: Bracket and brace balance (severity: error).

    Implements: REQ-CA-005, SCN-CA-005-02
    """
    pairs = [("{", "}"), ("(", ")"), ("[", "]")]
    details: list[str] = []

    for open_char, close_char in pairs:
        open_count = code.count(open_char)
        close_count = code.count(close_char)
        if open_count != close_count:
            details.append(
                f"'{open_char}': {open_count} vs '{close_char}': {close_count}"
            )

    if details:
        return CodeIssue(
            severity=CodeSeverity.ERROR,
            message=f"Unbalanced brackets: {'; '.join(details)}.",
            check_name="bracket_balance",
        )
    return None


def _check_host_only_apis(code: str) -> CodeIssue | None:
    """Check 5: No host-only API (severity: error).

    Implements: REQ-CA-005, SCN-CA-005-03
    """
    for pattern, api_name in _HOST_APIS:
        if pattern.search(code):
            return CodeIssue(
                severity=CodeSeverity.ERROR,
                message=(
                    f"Host-only API detected: {api_name}. "
                    "Kernel code must not contain host-side calls."
                ),
                check_name="host_only_api",
            )
    return None


def _check_dtype_match(code: str, dtype: str) -> CodeIssue | None:
    """Check 6: Kernel signature dtype match (severity: error).

    Uses heuristic matching — only clear mismatches are flagged. Errs on
    the side of permissiveness for complex type aliases.

    Implements: REQ-CA-005, SCN-CA-005-07
    """
    expected_types = DTYPE_MAP.get(dtype)
    if expected_types is None:
        # Unknown dtype — skip check (permissive)
        return None

    match = _GLOBAL_PARAMS_PATTERN.search(code)
    if not match:
        # No __global__ func found — check 1 will catch it
        return None

    params_str = match.group(1)

    # Check if any pointer parameter uses an expected type
    for expected_type in expected_types:
        if expected_type in params_str:
            return None

    # Also check for using/typedef aliases — if the code has a using/typedef
    # that maps to the expected type, accept it
    for expected_type in expected_types:
        # Check for typedef or using alias in the code body
        alias_pattern = re.compile(
            rf"(?:typedef|using)\s+\w+\s*=?\s*{re.escape(expected_type)}"
        )
        if alias_pattern.search(code):
            return None

    # Check if there are ANY pointer parameters at all
    if "*" not in params_str:
        # No pointer params — this might be an index-based kernel, skip check
        return None

    return CodeIssue(
        severity=CodeSeverity.ERROR,
        message=(
            f"Kernel signature dtype mismatch: expected "
            f"{expected_types[0]}* parameters for dtype={dtype}."
        ),
        check_name="dtype_match",
    )


def _check_nonempty_body(code: str) -> CodeIssue | None:
    """Check 7: Non-empty kernel body (severity: error).

    Extracts the body of the __global__ function and checks it contains
    at least one statement after stripping comments and whitespace.

    Implements: REQ-CA-005, SCN-CA-005-04
    """
    # Find the __global__ function and its opening brace
    global_match = re.search(
        r"__global__\s+(?:void|[\w:]+)\s+\w+\s*\([^)]*\)\s*\{",
        code,
    )
    if not global_match:
        # No __global__ func found — check 1 will catch it
        return None

    # Find the matching closing brace by counting
    start = global_match.end() - 1  # Position of the opening {
    depth = 1
    pos = start + 1
    while pos < len(code) and depth > 0:
        if code[pos] == "{":
            depth += 1
        elif code[pos] == "}":
            depth -= 1
        pos += 1

    if depth != 0:
        # Unbalanced braces — check 4 will catch it
        return None

    body = code[start + 1 : pos - 1]

    # Strip line comments
    body = re.sub(r"//[^\n]*", "", body)
    # Strip block comments
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.DOTALL)
    # Strip whitespace
    body = body.strip()

    if not body:
        return CodeIssue(
            severity=CodeSeverity.ERROR,
            message="Kernel body is empty or contains only comments.",
            check_name="empty_body",
        )
    return None


def validate_code(code: str, dtype: str) -> list[CodeIssue]:
    """Run all 7 validation checks on the given CUDA code.

    All checks always run (no short-circuit) so the full issue list is
    available for retry feedback.

    Args:
        code: The CUDA source code to validate.
        dtype: The expected data type from ProblemSpec.

    Returns:
        List of CodeIssue objects found. Empty list means no issues.

    Implements: REQ-CA-005
    """
    issues: list[CodeIssue] = []

    checks: list[CodeIssue | None] = [
        _check_global_function(code),
        _check_launch_bounds(code),
        _check_restrict_qualifiers(code),
        _check_bracket_balance(code),
        _check_host_only_apis(code),
        _check_dtype_match(code, dtype),
        _check_nonempty_body(code),
    ]

    for issue in checks:
        if issue is not None:
            issues.append(issue)

    return issues


def has_errors(issues: list[CodeIssue]) -> bool:
    """Check if any issue has error severity.

    Args:
        issues: List of code validation issues.

    Returns:
        True if any issue has error severity, False otherwise.
    """
    return any(issue.severity == CodeSeverity.ERROR for issue in issues)
