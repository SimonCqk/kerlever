"""Tests for the 6-layer CUDA optimization playbook.

Spec: docs/coding-agent/spec.md §6.2
"""

from __future__ import annotations

from kerlever.coding_agent.hardware import get_gpu_spec
from kerlever.coding_agent.playbook import (
    LAYER_1,
    format_playbook_layers,
    get_relevant_playbook,
)


class TestGetRelevantPlaybook:
    """Tests for playbook query function."""

    def test_memory_direction_includes_layer_2(self) -> None:
        """SCN-CA-003-01: Memory direction includes Layer 2."""
        gpu = get_gpu_spec("A100")
        layers = get_relevant_playbook("reduce_memory_bandwidth", gpu, "matmul")
        layer_numbers = [ly.layer_number for ly in layers]
        assert 1 in layer_numbers  # Always included
        assert 2 in layer_numbers  # Memory optimization

    def test_memory_direction_includes_coalesced_access(self) -> None:
        """SCN-CA-003-01: Memory layers include coalesced_access technique."""
        gpu = get_gpu_spec("A100")
        layers = get_relevant_playbook("reduce_memory_bandwidth", gpu, "matmul")
        layer_2 = [ly for ly in layers if ly.layer_number == 2]
        assert len(layer_2) == 1
        technique_names = [t.name for t in layer_2[0].techniques]
        assert "coalesced_access" in technique_names
        assert "shared_memory_tiling" in technique_names
        assert "vectorized_loads" in technique_names

    def test_compute_direction_includes_layer_3(self) -> None:
        """Compute direction includes Layer 3."""
        gpu = get_gpu_spec("A100")
        layers = get_relevant_playbook("increase_compute_throughput", gpu, "matmul")
        layer_numbers = [ly.layer_number for ly in layers]
        assert 3 in layer_numbers

    def test_advanced_direction_includes_layer_4(self) -> None:
        """Fusion/structural direction includes Layer 4."""
        gpu = get_gpu_spec("A100")
        layers = get_relevant_playbook("kernel_fusion", gpu, "matmul")
        layer_numbers = [ly.layer_number for ly in layers]
        assert 4 in layer_numbers

    def test_hopper_includes_layer_5_techniques(self) -> None:
        """SCN-CA-003-02: H100 includes Hopper-specific Layer 5."""
        gpu = get_gpu_spec("H100")
        layers = get_relevant_playbook("reduce_memory_bandwidth", gpu, "matmul")
        layer_5 = [ly for ly in layers if ly.layer_number == 5]
        assert len(layer_5) == 1
        technique_names = [t.name for t in layer_5[0].techniques]
        assert "hopper_tma" in technique_names
        assert "hopper_clusters" in technique_names
        assert "hopper_fp8" in technique_names
        # Should also have Ampere techniques since H100 supports cp_async
        assert "ampere_cp_async" in technique_names

    def test_ampere_includes_ampere_techniques(self) -> None:
        """A100 includes Ampere-specific Layer 5 but not Hopper."""
        gpu = get_gpu_spec("A100")
        layers = get_relevant_playbook("reduce_memory_bandwidth", gpu, "matmul")
        layer_5 = [ly for ly in layers if ly.layer_number == 5]
        assert len(layer_5) == 1
        technique_names = [t.name for t in layer_5[0].techniques]
        assert "ampere_cp_async" in technique_names
        assert "hopper_tma" not in technique_names

    def test_v100_no_layer_5(self) -> None:
        """V100 has no Layer 5 (no cp_async or TMA)."""
        gpu = get_gpu_spec("V100")
        layers = get_relevant_playbook("reduce_memory_bandwidth", gpu, "matmul")
        layer_5 = [ly for ly in layers if ly.layer_number == 5]
        assert len(layer_5) == 0

    def test_matmul_includes_layer_6(self) -> None:
        """SCN-CA-003-04: matmul op includes Layer 6 algorithm."""
        gpu = get_gpu_spec("A100")
        layers = get_relevant_playbook("optimize_compute", gpu, "matmul")
        layer_6 = [ly for ly in layers if ly.layer_number == 6]
        assert len(layer_6) == 1
        assert "matmul" in layer_6[0].techniques[0].applicable_when

    def test_reduction_includes_layer_6(self) -> None:
        """Reduction op includes Layer 6 algorithm."""
        gpu = get_gpu_spec("A100")
        layers = get_relevant_playbook("optimize_compute", gpu, "reduction")
        layer_6 = [ly for ly in layers if ly.layer_number == 6]
        assert len(layer_6) == 1
        assert layer_6[0].techniques[0].name == "reduction_warp_shuffle"

    def test_attention_includes_layer_6(self) -> None:
        """Attention op includes Layer 6 algorithm."""
        gpu = get_gpu_spec("A100")
        layers = get_relevant_playbook("optimize_compute", gpu, "attention")
        layer_6 = [ly for ly in layers if ly.layer_number == 6]
        assert len(layer_6) == 1
        assert layer_6[0].techniques[0].name == "attention_online_softmax"

    def test_unknown_direction_returns_layer_1_only(self) -> None:
        """SCN-CA-003-03: Unknown direction returns Layer 1 (universal fallback)."""
        gpu = get_gpu_spec("V100")  # V100 for no arch-specific layers
        layers = get_relevant_playbook(
            "some_unknown_optimization_tag", gpu, "unknown_op"
        )
        assert len(layers) >= 1
        assert layers[0].layer_number == 1

    def test_always_includes_layer_1(self) -> None:
        """INV-CA-004: Layer 1 is always included."""
        gpu = get_gpu_spec("A100")
        for direction in [
            "reduce_memory_bandwidth",
            "increase_compute_throughput",
            "kernel_fusion",
            "some_random_thing",
            "",
        ]:
            layers = get_relevant_playbook(direction, gpu, "matmul")
            assert layers[0].layer_number == 1

    def test_never_returns_empty(self) -> None:
        """INV-CA-004: Playbook query never returns empty."""
        gpu = get_gpu_spec("V100")
        layers = get_relevant_playbook("", gpu, "unknown")
        assert len(layers) > 0

    def test_multiple_direction_keywords_match_multiple_layers(self) -> None:
        """Direction with multiple keywords matches multiple layers."""
        gpu = get_gpu_spec("A100")
        # This direction contains both memory and compute keywords
        layers = get_relevant_playbook(
            "optimize_memory_and_compute_throughput", gpu, "matmul"
        )
        layer_numbers = [ly.layer_number for ly in layers]
        assert 2 in layer_numbers  # memory
        assert 3 in layer_numbers  # compute


class TestFormatPlaybookLayers:
    """Tests for playbook layer formatting."""

    def test_format_includes_layer_header(self) -> None:
        """Format includes layer number and name."""
        formatted = format_playbook_layers([LAYER_1])
        assert "Layer 1" in formatted
        assert "Block/Grid Configuration" in formatted

    def test_format_includes_technique_names(self) -> None:
        """Format includes technique names."""
        formatted = format_playbook_layers([LAYER_1])
        assert "block_size_tuning" in formatted
        assert "grid_sizing" in formatted
        assert "launch_bounds_declaration" in formatted

    def test_format_includes_technique_details(self) -> None:
        """Format includes technique details (when, gain, approach, caveats)."""
        formatted = format_playbook_layers([LAYER_1])
        assert "When:" in formatted
        assert "Expected gain:" in formatted
        assert "Approach:" in formatted
        assert "Caveats:" in formatted
