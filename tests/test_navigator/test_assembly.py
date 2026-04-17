"""Tests for Navigator Phase 4 directive assembly.

Spec: docs/navigator/spec.md §6.5
"""

from __future__ import annotations

from kerlever.navigator.assembly import assemble_directive
from kerlever.navigator.config import NavigatorConfig
from kerlever.navigator.types import GateResult, LLMDecision
from kerlever.types import (
    BaselineArtifact,
    CrossCandidateAnalysis,
    Mode,
    ObjectiveScore,
    OptimizationState,
    PerformanceObjective,
    ProblemSpec,
    RecombinationHint,
    ShapeBenchResult,
    ShapeCase,
    StaticAnalysis,
    SubMode,
    TabuEntry,
)


def _make_problem_spec() -> ProblemSpec:
    return ProblemSpec(
        op_name="matmul",
        op_semantics="C = A @ B",
        dtype="float32",
        target_gpu="A100",
        shape_cases=[ShapeCase(shape_id="s1", dims=[1024, 1024])],
        objective=PerformanceObjective(
            primary_metric="weighted_p50_us",
            aggregation="weighted_mean",
        ),
        target_metric_value=10.0,
        max_rounds=20,
        reference_kernel="__global__ void k() {}",
    )


def _make_baseline(
    kernel_hash: str = "abc123",
    score_value: float = 20.0,
) -> BaselineArtifact:
    return BaselineArtifact(
        kernel_hash=kernel_hash,
        source_code="__global__ void k() {}",
        compile_artifact=StaticAnalysis(),
        benchmark_results=[
            ShapeBenchResult(
                shape_id="s1",
                latency_p50_us=score_value,
                latency_p95_us=score_value * 1.1,
                run_count=10,
            ),
        ],
        objective_score=ObjectiveScore(
            metric_name="weighted_p50_us",
            value=score_value,
            relative_to_baseline=1.0,
            relative_to_incumbent=1.0,
        ),
    )


def _make_state(**kwargs: object) -> OptimizationState:
    ps = _make_problem_spec()
    bl = _make_baseline()
    defaults: dict[str, object] = {
        "problem_spec": ps,
        "baseline": bl,
        "incumbent": bl,
        "current_round": 3,
        "rounds": [],
        "attempts": [],
        "tabu_entries": [],
        "bottleneck_history": [],
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

    def test_legacy_fallback_preserves_hash_like_values(self) -> None:
        """Legacy simple hash-like winning genes remain usable as parents."""
        decision = LLMDecision(
            mode=Mode.EXPLORE,
            direction="recombine_top_kernels",
            sub_mode=SubMode.RECOMBINATION,
            reasoning="Combine winning traits",
            confidence="high",
        )
        cross_analysis = CrossCandidateAnalysis(
            insights=[],
            winning_genes=["hash_A", "hash_B"],
            recombination_suggestions=[],
        )
        state = _make_state()
        config = NavigatorConfig()

        directive = assemble_directive(decision, state, cross_analysis, config)

        assert directive.parent_candidates == ["hash_A", "hash_B"]

    def test_legacy_fallback_ignores_no_evidence_message(self) -> None:
        """Legacy no-evidence messages are not treated as parent hashes."""
        decision = LLMDecision(
            mode=Mode.EXPLORE,
            direction="recombine_top_kernels",
            sub_mode=SubMode.RECOMBINATION,
            reasoning="Combine winning traits",
            confidence="high",
        )
        cross_analysis = CrossCandidateAnalysis(
            insights=[],
            winning_genes=["No evidence-backed reusable genes identified."],
            recombination_suggestions=[],
        )
        state = _make_state()
        config = NavigatorConfig()

        directive = assemble_directive(decision, state, cross_analysis, config)

        assert directive.sub_mode == SubMode.DE_NOVO
        assert directive.parent_candidates is None
        assert directive.gene_map is None

    def test_structured_recombination_hint_is_preferred(self) -> None:
        """Structured hints set parents and gene_map before legacy fallback."""
        decision = LLMDecision(
            mode=Mode.EXPLORE,
            direction="recombine_top_kernels",
            sub_mode=SubMode.RECOMBINATION,
            reasoning="Combine structured genes",
            confidence="high",
        )
        cross_analysis = CrossCandidateAnalysis(
            insights=[],
            winning_genes=["legacy_A", "legacy_B"],
            recombination_suggestions=["legacy_section"],
            recombination_hints=[
                RecombinationHint(
                    hint_id="hint_1",
                    parent_candidates=["hash_A", "hash_B"],
                    gene_map={
                        "memory_access": "hash_A",
                        "compute_loop": "hash_B",
                    },
                    expected_benefit="Evidence-backed complementary genes.",
                    evidence_candidate_hashes=["hash_A", "hash_B"],
                    confidence="medium",
                )
            ],
        )
        state = _make_state()
        config = NavigatorConfig()

        directive = assemble_directive(decision, state, cross_analysis, config)

        assert directive.parent_candidates == ["hash_A", "hash_B"]
        assert directive.gene_map == {
            "memory_access": "hash_A",
            "compute_loop": "hash_B",
        }

    def test_structured_hint_filters_invalid_parent_hashes(self) -> None:
        """Structured recombination hints never pass unsafe parents onward."""
        decision = LLMDecision(
            mode=Mode.EXPLORE,
            direction="recombine_top_kernels",
            sub_mode=SubMode.RECOMBINATION,
            reasoning="Combine structured genes",
            confidence="high",
        )
        cross_analysis = CrossCandidateAnalysis(
            insights=[],
            winning_genes=["legacy_A", "legacy_B"],
            recombination_suggestions=[],
            recombination_hints=[
                RecombinationHint(
                    hint_id="hint_unsafe",
                    parent_candidates=["hash_A", "../secret", "hash_B"],
                    gene_map={
                        "memory_access": "hash_A",
                        "compute_loop": "hash_B",
                    },
                    expected_benefit="Evidence-backed complementary genes.",
                    evidence_candidate_hashes=["hash_A", "hash_B"],
                    confidence="medium",
                )
            ],
        )
        state = _make_state()
        config = NavigatorConfig()

        directive = assemble_directive(decision, state, cross_analysis, config)

        assert directive.sub_mode == SubMode.RECOMBINATION
        assert directive.parent_candidates == ["hash_A", "hash_B"]
        assert "../secret" not in directive.parent_candidates
        assert directive.gene_map == {
            "memory_access": "hash_A",
            "compute_loop": "hash_B",
        }


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
        # The tabu has this direction for a different base kernel hash
        incumbent = _make_baseline(kernel_hash="hash_B")
        state = _make_state(
            incumbent=incumbent,
            tabu_entries=[
                TabuEntry(
                    base_kernel_hash="hash_A",
                    direction="reduce_register_pressure",
                    sub_mode=None,
                    round_number=1,
                    expires_after_round=10,
                ),
            ],
        )
        config = NavigatorConfig()

        directive = assemble_directive(decision, state, None, config)

        # Should proceed (different hash)
        assert directive.direction == "reduce_register_pressure"
        # Should NOT have the tabu note in reason
        assert "tabu" not in directive.reason.lower()

    def test_same_direction_same_hash_noted(self) -> None:
        """Same direction on same base kernel is noted in reason."""
        decision = LLMDecision(
            mode=Mode.EXPLOIT,
            direction="reduce_register_pressure",
            reasoning="test",
            confidence="high",
        )
        state = _make_state(
            tabu_entries=[
                TabuEntry(
                    base_kernel_hash="abc123",
                    direction="reduce_register_pressure",
                    sub_mode=None,
                    round_number=1,
                    expires_after_round=10,
                ),
            ],
        )
        config = NavigatorConfig()

        directive = assemble_directive(decision, state, None, config)

        assert "tabu" in directive.reason.lower()


class TestTabuListOutput:
    """Tabu entries on directive contain only active entries."""

    def test_active_tabu_entries_only(self) -> None:
        decision = GateResult(
            mode=Mode.EXPLORE,
            direction="initial_exploration",
            reason="cold start",
            sub_mode=SubMode.DE_NOVO,
        )
        state = _make_state(
            current_round=5,
            tabu_entries=[
                TabuEntry(
                    base_kernel_hash="h1",
                    direction="d1",
                    sub_mode=None,
                    round_number=1,
                    expires_after_round=3,  # expired
                ),
                TabuEntry(
                    base_kernel_hash="h2",
                    direction="d2",
                    sub_mode=None,
                    round_number=2,
                    expires_after_round=7,  # active
                ),
                TabuEntry(
                    base_kernel_hash="h3",
                    direction="d3",
                    sub_mode=None,
                    round_number=3,
                    expires_after_round=5,  # active (exactly current round)
                ),
            ],
        )
        config = NavigatorConfig()

        directive = assemble_directive(decision, state, None, config)

        # Should only include active entries (expires_after_round >= 5)
        assert len(directive.tabu) == 2
        tabu_directions = {e.direction for e in directive.tabu}
        assert tabu_directions == {"d2", "d3"}


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
