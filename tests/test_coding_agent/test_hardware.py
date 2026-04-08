"""Tests for the GPU hardware constraint table and lookup.

Spec: docs/coding-agent/spec.md §6.1
"""

from __future__ import annotations

from kerlever.coding_agent.hardware import (
    GPU_SPECS,
    format_gpu_spec,
    get_gpu_spec,
)


class TestGetGpuSpec:
    """Tests for get_gpu_spec lookup function."""

    def test_a100_lookup(self) -> None:
        """SCN-CA-004-01: A100 returns correct specs."""
        spec = get_gpu_spec("A100")
        assert spec.arch == "sm_80"
        assert spec.smem_per_sm_kb == 164
        assert spec.max_smem_per_block_kb == 163
        assert spec.registers_per_sm == 65536
        assert spec.max_registers_per_thread == 255
        assert spec.max_warps_per_sm == 64
        assert spec.max_threads_per_block == 1024
        assert spec.hbm_bandwidth_tbps == 2.0
        assert spec.l2_cache_mb == 40
        assert spec.supports_cp_async is True
        assert spec.supports_tma is False
        assert spec.supports_fp8 is False
        assert "fp16" in spec.tensor_core_types
        assert "bf16" in spec.tensor_core_types
        assert "tf32" in spec.tensor_core_types

    def test_h100_lookup(self) -> None:
        """SCN-CA-004-02: H100 returns correct specs."""
        spec = get_gpu_spec("H100")
        assert spec.arch == "sm_90"
        assert spec.smem_per_sm_kb == 228
        assert spec.max_smem_per_block_kb == 227
        assert spec.supports_cp_async is True
        assert spec.supports_tma is True
        assert spec.supports_fp8 is True
        assert "fp8" in spec.tensor_core_types

    def test_v100_lookup(self) -> None:
        """V100 returns correct specs."""
        spec = get_gpu_spec("V100")
        assert spec.arch == "sm_70"
        assert spec.smem_per_sm_kb == 96
        assert spec.supports_cp_async is False
        assert spec.supports_tma is False
        assert spec.supports_fp8 is False

    def test_case_insensitive_lookup(self) -> None:
        """Lookup is case-insensitive."""
        spec = get_gpu_spec("a100")
        assert spec.arch == "sm_80"

        spec = get_gpu_spec("h100")
        assert spec.arch == "sm_90"

    def test_whitespace_stripped(self) -> None:
        """Lookup strips whitespace."""
        spec = get_gpu_spec("  A100  ")
        assert spec.arch == "sm_80"

    def test_unknown_gpu_returns_defaults(self) -> None:
        """SCN-CA-004-03: Unknown GPU returns conservative defaults."""
        spec = get_gpu_spec("RTX_9090")
        assert spec.arch == "sm_70"
        assert spec.smem_per_sm_kb == 48
        assert spec.max_smem_per_block_kb == 48
        assert spec.max_registers_per_thread == 255
        assert spec.max_threads_per_block == 1024
        assert spec.supports_cp_async is False
        assert spec.supports_tma is False
        assert spec.supports_fp8 is False

    def test_empty_string_returns_defaults(self) -> None:
        """Empty GPU name returns defaults."""
        spec = get_gpu_spec("")
        assert spec.smem_per_sm_kb == 48
        assert spec.supports_cp_async is False

    def test_gpu_specs_table_has_required_gpus(self) -> None:
        """GPU_SPECS table contains at least V100, A100, H100."""
        assert "V100" in GPU_SPECS
        assert "A100" in GPU_SPECS
        assert "H100" in GPU_SPECS


class TestFormatGpuSpec:
    """Tests for GPU spec formatting."""

    def test_format_includes_architecture(self) -> None:
        """Format includes architecture info."""
        spec = get_gpu_spec("A100")
        formatted = format_gpu_spec(spec)
        assert "sm_80" in formatted

    def test_format_includes_smem(self) -> None:
        """Format includes shared memory info."""
        spec = get_gpu_spec("A100")
        formatted = format_gpu_spec(spec)
        assert "164" in formatted

    def test_format_cp_async_supported(self) -> None:
        """Format shows cp.async as supported for A100."""
        spec = get_gpu_spec("A100")
        formatted = format_gpu_spec(spec)
        assert "Supported" in formatted

    def test_format_tma_not_available(self) -> None:
        """Format shows TMA as not available for A100."""
        spec = get_gpu_spec("A100")
        formatted = format_gpu_spec(spec)
        assert "TMA: Not available" in formatted
