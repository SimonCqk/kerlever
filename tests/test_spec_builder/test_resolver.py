"""Tests for reference kernel resolution.

Spec: docs/spec-builder/spec.md §6.1
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from kerlever.spec_builder.resolver import resolve_reference_kernel

_VALID_CUDA = """\
__global__ void matmul(const half* A, const half* B, half* C, int M, int N, int K) {
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


class TestInlineResolution:
    """Inline CUDA source passes through with marker check."""

    @pytest.mark.asyncio
    async def test_inline_with_global_passes(self) -> None:
        result = await resolve_reference_kernel(_VALID_CUDA)
        assert "__global__" in result

    @pytest.mark.asyncio
    async def test_inline_with_device_passes(self) -> None:
        source = "__device__ float helper(float x) { return x * x; }"
        result = await resolve_reference_kernel(source)
        assert "__device__" in result

    @pytest.mark.asyncio
    async def test_inline_without_cuda_markers_fails(self) -> None:
        with pytest.raises(ValueError, match="not appear to be valid CUDA"):
            await resolve_reference_kernel("void foo() { return; }")


class TestFileResolution:
    """file:// URI resolution."""

    @pytest.mark.asyncio
    async def test_file_resolves(self, tmp_path: pytest.TempPathFactory) -> None:
        kernel_file = tmp_path / "kernel.cu"  # type: ignore[operator]
        kernel_file.write_text(_VALID_CUDA, encoding="utf-8")
        result = await resolve_reference_kernel(f"file://{kernel_file}")
        assert "__global__" in result

    @pytest.mark.asyncio
    async def test_missing_file_errors(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            await resolve_reference_kernel("file:///nonexistent/path.cu")

    @pytest.mark.asyncio
    async def test_file_without_cuda_markers_errors(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        kernel_file = tmp_path / "plain.c"  # type: ignore[operator]
        kernel_file.write_text("void foo() { return; }", encoding="utf-8")
        with pytest.raises(ValueError, match="not appear to be valid CUDA"):
            await resolve_reference_kernel(f"file://{kernel_file}")


class TestUrlResolution:
    """https:// URL resolution (mocked)."""

    @pytest.mark.asyncio
    async def test_url_resolves(self) -> None:
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = _VALID_CUDA.encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        target = "kerlever.spec_builder.resolver.urllib.request.urlopen"
        with patch(target, return_value=mock_response):
            result = await resolve_reference_kernel("https://example.com/kernel.cu")
            assert "__global__" in result

    @pytest.mark.asyncio
    async def test_url_fetch_fail_errors(self) -> None:
        import urllib.error

        target = "kerlever.spec_builder.resolver.urllib.request.urlopen"
        with (
            patch(
                target,
                side_effect=urllib.error.HTTPError(
                    "https://example.com/404.cu",
                    404,
                    "Not Found",
                    {},  # type: ignore[arg-type]
                    None,
                ),
            ),
            pytest.raises(ValueError, match="status 404"),
        ):
            await resolve_reference_kernel("https://example.com/404.cu")
