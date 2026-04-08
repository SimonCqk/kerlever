"""Tests for Navigator Phase 4 directive assembly.

Spec: docs/navigator/spec.md §6.5
"""

from __future__ import annotations

from kerlever.navigator.assembly import assemble_directive
from kerlever.navigator.config import NavigatorConfig
from kerlever.navigator.types import GateResult, LLMDecision
from kerlever.types import (
    CrossCandidateAnalysis,
    Mode,
    OptimizationState,
    ProblemSpec,
    SubMode,
)


def _make_problem_spec() -> ProblemSpec:
    return ProblemSpec(
        op_name="matmul",
        op_semantics="C = A @ B",
        shapes=[[1024, 1024], [1024, 1024]],
        dtype="float32",
        target_gpu="A100",
        baseline_perf_us=100.0,
        target_perf_us=10.0,
        tolerance=0.05,
        max_rounds=20,
        reference_kernel="__global__ void k() {}",
    )


def _make_state(**kwargs: object) -> OptimizationState:
    defaults: dict[str, object] = {
        "problem_spec": _make_problem_spec(),
        "current_round": 3,
        "global_best_hash": "abc123",
        "global_best_latency_us": 20.0,
    }
    defaults.update(kwargs)
    return OptimizationState(**defaults)  # type: ignore[arg-type]


class TestExploitDirective:
    """EXPLOIT mode directive assembly."""

    def test_exploit_sets_base_kernel_hash(self) -> None:
        decision = GateResult(
            mode=Mode.EXPLOIT,
            direction="fine_tune",
            reason="near target",
            sub_mode=SubMode.PARAM_SEARCH,
        )
        state = _make_state()
        config = NavigatorConfig(exploit_candidates=5)

        directive = assemble_directive(decision, state, None, config)

        assert directive.mode == Mode.EXPLOIT
        assert directive.base_kernel_hash == "abc123"
        assert directive.num_candidates == 5

    def test_exploit_infers_sub_mode(self) -> None:
        decision = LLMDecision(
            mode=Mode.EXPLOIT,
            direction="reduce_memory_bandwidth",
            reasoning="test",
            confidence="high",
        )
        state = _make_state()
        config = NavigatorConfig()

        directive = assemble_directive(decision, state, None, config)

        assert directive.sub_mode is not None
        # Should infer LOCAL_REWRITE for generic direction
        assert directive.sub_mode == SubMode.LOCAL_REWRITE

    def test_exploit_candidates_from_config(self) -> None:
        decision = GateResult(
            mode=Mode.EXPLOIT,
            direction="fine_tune",
            reason="near target",
            sub_mode=SubMode.PARAM_SEARCH,
        )
        state = _make_state()
        config = NavigatorConfig(exploit_candidates=7)

        directive = assemble_directive(decision, state, None, config)

        assert directive.num_candidates == 7


class TestExploreDeNovoDirective:
    """EXPLORE DE_NOVO mode directive assembly."""

    def test_de_novo_base_hash_none(self) -> None:
        decision = GateResult(
            mode=Mode.EXPLORE,
            direction="initial_exploration",
            reason="cold start",
            sub_mode=SubMode.DE_NOVO,
        )
        state = _make_state()
        config = NavigatorConfig(explore_candidates=3)

        directive = assemble_directive(decision, state, None, config)

        assert directive.mode == Mode.EXPLORE
        assert directive.sub_mode == SubMode.DE_NOVO
        assert directive.base_kernel_hash is None
        assert directive.num_candidates == 3
        assert directive.parent_candidates is None
        assert directive.gene_map is None

    def test_explore_candidates_from_config(self) -> None:
        decision = GateResult(
            mode=Mode.EXPLORE,
            direction="structural_change",
            reason="plateau",
        )
        state = _make_state()
        config = NavigatorConfig(explore_candidates=2)

        directive = assemble_directive(decision, state, None, config)

        assert directive.num_candidates == 2


class TestExploreRecombinationDirective:
    """EXPLORE RECOMBINATION mode directive assembly."""

    def test_recombination_sets_parents(self) -> None:
        decision = LLMDecision(
            mode=Mode.EXPLORE,
            direction="recombine_top_kernels",
            sub_mode=SubMode.RECOMBINATION,
            reasoning="Combine winning traits",
            confidence="high",
        )
        cross_analysis = CrossCandidateAnalysis(
            insights=["shared memory usage wins"],
            winning_genes=["hash_A", "hash_B"],
            recombination_suggestions=[
                "memory_pattern_from_A",
                "compute_pattern_from_B",
            ],
        )
        state = _make_state()
        config = NavigatorConfig()

        directive = assemble_directive(decision, state, cross_analysis, config)

        assert directive.mode == Mode.EXPLORE
        assert directive.sub_mode == SubMode.RECOMBINATION
        assert directive.base_kernel_hash is None
        assert directive.parent_candidates is not None
        assert len(directive.parent_candidates) >= 2
        assert directive.gene_map is not None


class TestTabuFilter:
    """Tabu filtering matches on (base_hash, direction) pair."""

    def test_same_direction_different_hash_allowed(self) -> None:
        """Same direction on a different base kernel is allowed."""
        decision = LLMDecision(
            mode=Mode.EXPLOIT,
            direction="reduce_register_pressure",
            reasoning="test",
            confidence="high",
        )
        # The tabu list has this direction from a recent round
        # but the base kernel is different
        state = _make_state(
            global_best_hash="hash_B",
            decision_log=[
                {
                    "round_number": 1,
                    "directive": {
                        "mode": "EXPLOIT",
                        "direction": "reduce_register_pressure",
                        "reason": "test",
                        "num_candidates": 3,
                    },
                }
            ],
        )
        config = NavigatorConfig()

        directive = assemble_directive(decision, state, None, config)

        # Should proceed (different hash) — but the current
        # implementation notes in reason when tabu triggers
        assert directive.direction == "reduce_register_pressure"


class TestTabuListOutput:
    """Tabu list on directive is windowed to last W rounds."""

    def test_tabu_windowed(self) -> None:
        decision = GateResult(
            mode=Mode.EXPLORE,
            direction="initial_exploration",
            reason="cold start",
            sub_mode=SubMode.DE_NOVO,
        )
        state = _make_state(
            tabu_list=["a", "b", "c", "d", "e", "f", "g"],
        )
        config = NavigatorConfig(tabu_window=5)

        directive = assemble_directive(decision, state, None, config)

        # Should only include the last 5 entries
        assert directive.tabu == ["c", "d", "e", "f", "g"]


class TestHardConstraints:
    """Hard constraints are always populated."""

    def test_exploit_has_hard_constraints(self) -> None:
        decision = GateResult(
            mode=Mode.EXPLOIT,
            direction="fine_tune",
            reason="near target",
            sub_mode=SubMode.PARAM_SEARCH,
        )
        state = _make_state()
        config = NavigatorConfig()

        directive = assemble_directive(decision, state, None, config)

        assert directive.hard_constraints is not None
        assert len(directive.hard_constraints) > 0

    def test_explore_has_hard_constraints(self) -> None:
        decision = GateResult(
            mode=Mode.EXPLORE,
            direction="structural_change",
            reason="plateau",
            sub_mode=SubMode.DE_NOVO,
        )
        state = _make_state()
        config = NavigatorConfig()

        directive = assemble_directive(decision, state, None, config)

        assert directive.hard_constraints is not None
