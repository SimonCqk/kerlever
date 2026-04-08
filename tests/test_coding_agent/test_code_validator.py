"""Tests for the CUDA code validator — 7 regex-level checks.

Spec: docs/coding-agent/spec.md §6.4
"""

from __future__ import annotations

from kerlever.coding_agent.code_validator import has_errors, validate_code
from kerlever.coding_agent.types import CodeSeverity

# A well-formed kernel for baseline testing
VALID_KERNEL = """\
__launch_bounds__(256, 2)
__global__ void matmul_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        half sum = 0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
"""


class TestCheck1GlobalFunction:
    """Check 1: __global__ function exists (error)."""

    def test_missing_global_is_error(self) -> None:
        """SCN-CA-005-01: Missing __global__ is rejected."""
        code = "void some_function(float* a) { a[0] = 1.0f; }"
        issues = validate_code(code, "float32")
        errors = [i for i in issues if i.severity == CodeSeverity.ERROR]
        assert any("__global__" in e.message for e in errors)
        assert has_errors(issues)

    def test_global_present_passes(self) -> None:
        """__global__ function present passes check 1."""
        issues = validate_code(VALID_KERNEL, "float16")
        global_errors = [
            i
            for i in issues
            if i.check_name == "global_function" and i.severity == CodeSeverity.ERROR
        ]
        assert len(global_errors) == 0


class TestCheck2LaunchBounds:
    """Check 2: __launch_bounds__ present (warning)."""

    def test_missing_launch_bounds_is_warning(self) -> None:
        """SCN-CA-005-05: Missing __launch_bounds__ produces warning."""
        code = """\
__global__ void kernel(half* __restrict__ out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = (half)1.0f;
}
"""
        issues = validate_code(code, "float16")
        warnings = [i for i in issues if i.severity == CodeSeverity.WARNING]
        assert any("__launch_bounds__" in w.message for w in warnings)
        # Warning should NOT cause rejection
        errors = [i for i in issues if i.severity == CodeSeverity.ERROR]
        assert not any(e.check_name == "launch_bounds" for e in errors)

    def test_launch_bounds_present_no_warning(self) -> None:
        """__launch_bounds__ present produces no check-2 warning."""
        issues = validate_code(VALID_KERNEL, "float16")
        lb_issues = [i for i in issues if i.check_name == "launch_bounds"]
        assert len(lb_issues) == 0


class TestCheck3RestrictQualifier:
    """Check 3: __restrict__ on pointer parameters (warning)."""

    def test_missing_restrict_is_warning(self) -> None:
        """SCN-CA-005-06: Missing __restrict__ produces warning."""
        code = """\
__launch_bounds__(256, 2)
__global__ void kernel(half* A, half* B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) B[idx] = A[idx];
}
"""
        issues = validate_code(code, "float16")
        warnings = [i for i in issues if i.severity == CodeSeverity.WARNING]
        assert any("__restrict__" in w.message for w in warnings)

    def test_restrict_present_no_warning(self) -> None:
        """All pointers with __restrict__ produces no check-3 warning."""
        issues = validate_code(VALID_KERNEL, "float16")
        restrict_issues = [i for i in issues if i.check_name == "restrict_qualifier"]
        assert len(restrict_issues) == 0


class TestCheck4BracketBalance:
    """Check 4: Bracket and brace balance (error)."""

    def test_unbalanced_braces_is_error(self) -> None:
        """SCN-CA-005-02: Unbalanced braces are rejected."""
        code = """\
__launch_bounds__(256, 2)
__global__ void kernel(half* __restrict__ a, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        a[idx] = (half)1.0f;
    // Missing closing brace
}
"""
        issues = validate_code(code, "float16")
        assert has_errors(issues)
        bracket_errors = [i for i in issues if i.check_name == "bracket_balance"]
        assert len(bracket_errors) == 1
        assert "Unbalanced" in bracket_errors[0].message

    def test_balanced_braces_passes(self) -> None:
        """Balanced braces pass check 4."""
        issues = validate_code(VALID_KERNEL, "float16")
        bracket_errors = [i for i in issues if i.check_name == "bracket_balance"]
        assert len(bracket_errors) == 0


class TestCheck5HostOnlyAPI:
    """Check 5: No host-only API (error)."""

    def test_malloc_detected(self) -> None:
        """SCN-CA-005-03: malloc detected as host-only API."""
        code = """\
__launch_bounds__(256, 2)
__global__ void kernel(half* __restrict__ a, int N) {
    int* temp = (int*)malloc(sizeof(int) * N);
    a[0] = (half)1.0f;
}
"""
        issues = validate_code(code, "float16")
        assert has_errors(issues)
        host_errors = [i for i in issues if i.check_name == "host_only_api"]
        assert len(host_errors) == 1
        assert "malloc" in host_errors[0].message

    def test_printf_detected(self) -> None:
        """printf detected as host-only API."""
        code = """\
__launch_bounds__(256, 2)
__global__ void kernel(half* __restrict__ a, int N) {
    printf("hello");
    a[0] = (half)1.0f;
}
"""
        issues = validate_code(code, "float16")
        host_errors = [i for i in issues if i.check_name == "host_only_api"]
        assert len(host_errors) == 1
        assert "printf" in host_errors[0].message

    def test_std_detected(self) -> None:
        """SCN-CA-005-03: std:: detected as host-only API."""
        code = """\
__launch_bounds__(256, 2)
__global__ void kernel(half* __restrict__ a, int N) {
    std::vector<int> v;
    a[0] = (half)1.0f;
}
"""
        issues = validate_code(code, "float16")
        host_errors = [i for i in issues if i.check_name == "host_only_api"]
        assert len(host_errors) == 1
        assert "std::" in host_errors[0].message

    def test_cuda_malloc_detected(self) -> None:
        """cudaMalloc detected as host-only API."""
        code = """\
__launch_bounds__(256, 2)
__global__ void kernel(half* __restrict__ a, int N) {
    half* temp;
    cudaMalloc(&temp, N);
    a[0] = (half)1.0f;
}
"""
        issues = validate_code(code, "float16")
        host_errors = [i for i in issues if i.check_name == "host_only_api"]
        assert len(host_errors) == 1

    def test_no_host_api_passes(self) -> None:
        """Clean kernel passes check 5."""
        issues = validate_code(VALID_KERNEL, "float16")
        host_errors = [i for i in issues if i.check_name == "host_only_api"]
        assert len(host_errors) == 0


class TestCheck6DtypeMatch:
    """Check 6: Kernel signature dtype match (error)."""

    def test_dtype_mismatch_float32_for_float16(self) -> None:
        """SCN-CA-005-07: float* when float16 expected is rejected."""
        code = """\
__launch_bounds__(256, 2)
__global__ void kernel(float* __restrict__ A, float* __restrict__ B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) B[idx] = A[idx];
}
"""
        issues = validate_code(code, "float16")
        dtype_errors = [i for i in issues if i.check_name == "dtype_match"]
        assert len(dtype_errors) == 1
        assert "half" in dtype_errors[0].message

    def test_correct_dtype_passes(self) -> None:
        """Correct dtype (half for float16) passes."""
        issues = validate_code(VALID_KERNEL, "float16")
        dtype_errors = [i for i in issues if i.check_name == "dtype_match"]
        assert len(dtype_errors) == 0

    def test_float32_correct(self) -> None:
        """float* for float32 passes."""
        code = """\
__launch_bounds__(256, 2)
__global__ void kernel(float* __restrict__ A, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) A[idx] = 1.0f;
}
"""
        issues = validate_code(code, "float32")
        dtype_errors = [i for i in issues if i.check_name == "dtype_match"]
        assert len(dtype_errors) == 0

    def test_unknown_dtype_skipped(self) -> None:
        """Unknown dtype skips check (permissive)."""
        code = """\
__launch_bounds__(256, 2)
__global__ void kernel(float* __restrict__ A, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) A[idx] = 1.0f;
}
"""
        issues = validate_code(code, "custom_type_99")
        dtype_errors = [i for i in issues if i.check_name == "dtype_match"]
        assert len(dtype_errors) == 0


class TestCheck7NonEmptyBody:
    """Check 7: Non-empty kernel body (error)."""

    def test_empty_body_is_error(self) -> None:
        """SCN-CA-005-04: Empty kernel body is rejected."""
        code = """\
__launch_bounds__(256, 2)
__global__ void kernel(half* __restrict__ a, int N) {
}
"""
        issues = validate_code(code, "float16")
        assert has_errors(issues)
        body_errors = [i for i in issues if i.check_name == "empty_body"]
        assert len(body_errors) == 1

    def test_comments_only_body_is_error(self) -> None:
        """Body with only comments is rejected."""
        code = """\
__launch_bounds__(256, 2)
__global__ void kernel(half* __restrict__ a, int N) {
    // This is just a comment
    /* Another comment */
}
"""
        issues = validate_code(code, "float16")
        body_errors = [i for i in issues if i.check_name == "empty_body"]
        assert len(body_errors) == 1

    def test_non_empty_body_passes(self) -> None:
        """Non-empty body passes check 7."""
        issues = validate_code(VALID_KERNEL, "float16")
        body_errors = [i for i in issues if i.check_name == "empty_body"]
        assert len(body_errors) == 0


class TestValidCodePassesAll:
    """SCN-CA-005-08: Valid code passes all checks."""

    def test_valid_kernel_no_errors(self) -> None:
        """Well-formed kernel has no error-severity issues."""
        issues = validate_code(VALID_KERNEL, "float16")
        assert not has_errors(issues)

    def test_all_checks_run(self) -> None:
        """All 7 checks are exercised on valid code (no short-circuit)."""
        # This validates the structure — each check should have run
        issues = validate_code(VALID_KERNEL, "float16")
        # For valid code, we expect zero errors and zero warnings
        assert len(issues) == 0
