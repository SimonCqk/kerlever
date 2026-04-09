"""Kerlever CLI entry point — run via `python -m kerlever <spec.yaml> <workdir>`.

Spec: docs/orchestrator/spec.md
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from kerlever.orchestrator import Orchestrator
from kerlever.problem_spec import load_problem_spec
from kerlever.stubs import (
    StubCodingAgent,
    StubCrossCandidateAnalyzer,
    StubGPUPipeline,
    StubStrategyNavigator,
)


def main() -> None:
    """CLI entry point: load spec, create stubs, run orchestrator."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if len(sys.argv) != 3:
        print(
            "Usage: python -m kerlever <spec.yaml> <workdir>",
            file=sys.stderr,
        )
        sys.exit(1)

    spec_path = Path(sys.argv[1])
    workdir = Path(sys.argv[2])

    problem_spec = load_problem_spec(spec_path)

    orchestrator = Orchestrator(
        problem_spec=problem_spec,
        strategy_navigator=StubStrategyNavigator(),
        coding_agent=StubCodingAgent(),
        gpu_pipeline=StubGPUPipeline(),
        cross_analyzer=StubCrossCandidateAnalyzer(),
        workdir=workdir,
    )

    result = asyncio.run(orchestrator.run())

    print(f"Status: {result.status}")
    print(f"Best kernel hash: {result.best_kernel_hash}")
    if result.best_objective_score is not None:
        print(f"Best objective score: {result.best_objective_score:.3f}")
    else:
        print("Best objective score: N/A")
    print(f"Total rounds: {result.total_rounds}")
    print(f"Total candidates evaluated: {result.total_candidates_evaluated}")


if __name__ == "__main__":
    main()
