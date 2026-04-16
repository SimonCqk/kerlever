"""Strategy Navigator — search-space controller for the optimization loop.

Public API: ``StrategyNavigator`` class satisfying ``StrategyNavigatorProtocol``.

Spec: docs/navigator/spec.md
"""

from __future__ import annotations

import logging

from kerlever.llm_client import LLMClientProtocol
from kerlever.navigator.assembly import assemble_directive
from kerlever.navigator.config import NavigatorConfig
from kerlever.navigator.gates import check_gates
from kerlever.navigator.llm_reasoning import run_llm_reasoning
from kerlever.navigator.signals import compute_derived_signals
from kerlever.navigator.types import DerivedSignals, LLMDecision
from kerlever.navigator.ucb1 import compute_direction_stats, ucb1_select
from kerlever.types import (
    CrossCandidateAnalysis,
    Mode,
    OptimizationState,
    ProblemSpec,
    RoundSummary,
    StrategyDirective,
    SubMode,
    TabuEntry,
)

__all__ = ["StrategyNavigator"]

logger = logging.getLogger(__name__)


class StrategyNavigator:
    """Search-space controller that decides what to optimize next.

    Combines deterministic gates for clear signals with LLM reasoning
    for ambiguous situations. LLM failures degrade gracefully to UCB1.

    Satisfies ``StrategyNavigatorProtocol`` from ``kerlever.protocols``.

    Implements: REQ-NAV-001 through REQ-NAV-009
    """

    def __init__(
        self,
        llm_client: LLMClientProtocol | None = None,
        config: NavigatorConfig | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._config = config or NavigatorConfig()

    async def decide(
        self,
        problem_spec: ProblemSpec,
        optimization_state: OptimizationState,
        round_summary: RoundSummary | None,
        cross_analysis: CrossCandidateAnalysis | None,
    ) -> StrategyDirective:
        """Decide the strategy directive for the next round.

        Follows the 5-phase flow:
        1. Compute derived signals
        2. Check deterministic gates
        3. LLM reasoning (or UCB1 fallback)
        4. Assemble directive
        5. Return directive

        Args:
            problem_spec: Target problem specification.
            optimization_state: Full accumulated optimization state.
            round_summary: Previous round summary (None on round 0).
            cross_analysis: Cross-candidate analysis (None on round 0).

        Returns:
            Complete StrategyDirective for the Coding Agent.

        Implements: REQ-NAV-001 through REQ-NAV-009
        Invariant: INV-NAV-003 (LLM failures never stall the system)
        Invariant: INV-NAV-006 (every directive has valid mode and non-empty reason)
        """
        # Phase 1: Compute derived signals
        try:
            signals = compute_derived_signals(optimization_state, self._config)
        except Exception:
            logger.warning(
                "Signal computation failed, using safe defaults",
                exc_info=True,
            )
            signals = DerivedSignals(
                avg_delta=0.0,
                is_plateau=False,
                is_regress=False,
                stable_bottleneck=None,
                new_bottleneck=None,
                consecutive_exploit_rounds=0,
                direction_attempt_counts={},
                exhausted_directions=set(),
            )

        # Phase 2: Check deterministic gates
        try:
            gate_result = check_gates(
                signals, optimization_state, problem_spec, self._config
            )
        except Exception:
            logger.warning(
                "Gate evaluation failed, proceeding to Phase 3",
                exc_info=True,
            )
            gate_result = None

        if gate_result is not None:
            # Gate matched — skip to Phase 4
            try:
                return assemble_directive(
                    gate_result, optimization_state, cross_analysis, self._config
                )
            except Exception:
                logger.warning(
                    "Assembly failed for gate result, returning safe directive",
                    exc_info=True,
                )
                return _safe_directive(optimization_state, self._config)

        # Phase 3: LLM reasoning (or UCB1 if no LLM client)
        decision: LLMDecision | None = None

        if self._llm_client is not None:
            try:
                decision = await run_llm_reasoning(
                    optimization_state,
                    signals,
                    cross_analysis,
                    self._llm_client,
                    self._config,
                )
            except Exception:
                logger.warning(
                    "LLM reasoning failed after retries, falling back to UCB1",
                    exc_info=True,
                )

        # UCB1 fallback (when LLM failed or no LLM client)
        if decision is None:
            decision = _ucb1_fallback(optimization_state, signals, self._config)

        # Phase 4: Assemble directive
        try:
            return assemble_directive(
                decision, optimization_state, cross_analysis, self._config
            )
        except Exception:
            logger.warning(
                "Assembly failed for LLM/UCB1 decision, returning safe directive",
                exc_info=True,
            )
            return _safe_directive(optimization_state, self._config)


def _ucb1_fallback(
    state: OptimizationState,
    signals: DerivedSignals,
    config: NavigatorConfig,
) -> LLMDecision:
    """Produce an LLMDecision using UCB1 direction selection.

    Used when LLM reasoning fails or no LLM client is available.
    Per spec §6.4, UCB1 fallback produces EXPLOIT mode.
    """
    stats = compute_direction_stats(state)
    direction = ucb1_select(
        stats, state.current_round, signals.exhausted_directions, config
    )

    return LLMDecision(
        mode=Mode.EXPLOIT,
        direction=direction,
        sub_mode=None,
        reasoning=("UCB1 fallback: selected direction based on upper confidence bound"),
        confidence="medium",
    )


def _safe_directive(
    state: OptimizationState,
    config: NavigatorConfig,
) -> StrategyDirective:
    """Return a minimal safe directive when assembly errors occur.

    Per spec §6.6 Error Handling: EXPLORE mode, DE_NOVO, generic direction.
    """
    active_tabu: list[TabuEntry] = [
        e for e in state.tabu_entries if e.expires_after_round >= state.current_round
    ]
    return StrategyDirective(
        mode=Mode.EXPLORE,
        direction="initial_exploration",
        reason="Safe fallback: assembly error occurred",
        base_kernel_hash=None,
        num_candidates=config.explore_candidates,
        tabu=active_tabu,
        sub_mode=SubMode.DE_NOVO,
    )
