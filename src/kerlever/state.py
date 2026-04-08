"""Kerlever state — StateManager for workdir I/O with atomic writes.

Spec: docs/orchestrator/spec.md

All state file writes use write-to-temporary-then-rename pattern
to ensure atomicity (INV-ORCH-005).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from kerlever.types import (
    OptimizationResult,
    OptimizationState,
    RoundState,
)

logger = logging.getLogger(__name__)


class StateManager:
    """Manages all workdir filesystem operations for the Orchestrator.

    Creates the workdir directory structure on init and provides methods
    for atomically persisting state, round data, kernel source, decision
    log entries, and the final result.

    Invariant: INV-ORCH-005 (atomic writes via tmp+rename)
    """

    def __init__(self, workdir: Path) -> None:
        self._workdir = workdir
        self._rounds_dir = workdir / "rounds"
        self._kernels_dir = workdir / "kernels"
        self._init_workdir()

    def _init_workdir(self) -> None:
        """Create workdir and subdirectories if they do not exist."""
        self._workdir.mkdir(parents=True, exist_ok=True)
        self._rounds_dir.mkdir(exist_ok=True)
        self._kernels_dir.mkdir(exist_ok=True)

        state_path = self._workdir / "state.json"
        if state_path.exists():
            logger.warning(
                "Existing state.json found in workdir %s; starting fresh (V1)",
                self._workdir,
            )

    def _atomic_write(self, path: Path, data: str) -> None:
        """Write data to a file atomically using tmp+rename.

        Implements: INV-ORCH-005
        """
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(data, encoding="utf-8")
        os.replace(tmp_path, path)

    def save_state(self, state: OptimizationState) -> None:
        """Save the full optimization state snapshot to state.json.

        Implements: REQ-ORCH-002
        Invariant: INV-ORCH-005 (atomic write)
        """
        path = self._workdir / "state.json"
        self._atomic_write(path, state.model_dump_json(indent=2))

    def load_state(self) -> OptimizationState | None:
        """Load optimization state from state.json, or None if not found."""
        path = self._workdir / "state.json"
        if not path.exists():
            return None
        data = path.read_text(encoding="utf-8")
        return OptimizationState.model_validate_json(data)

    def save_round(self, round_state: RoundState) -> None:
        """Save a round's complete state to rounds/round_NNN.json.

        Implements: REQ-ORCH-002
        Invariant: INV-ORCH-005 (atomic write)
        """
        filename = f"round_{round_state.round_number:03d}.json"
        path = self._rounds_dir / filename
        self._atomic_write(path, round_state.model_dump_json(indent=2))

    def load_round(self, round_number: int) -> RoundState | None:
        """Load a specific round's state, or None if not found."""
        filename = f"round_{round_number:03d}.json"
        path = self._rounds_dir / filename
        if not path.exists():
            return None
        data = path.read_text(encoding="utf-8")
        return RoundState.model_validate_json(data)

    def save_kernel(self, code_hash: str, source_code: str) -> None:
        """Save a kernel candidate's source code to kernels/<hash>.cu.

        Implements: INV-ORCH-003 (persist before evaluation)
        """
        path = self._kernels_dir / f"{code_hash}.cu"
        path.write_text(source_code, encoding="utf-8")

    def append_decision(self, entry: dict[str, object]) -> None:
        """Append a decision log entry to decision_log.jsonl.

        Implements: REQ-ORCH-002
        """
        path = self._workdir / "decision_log.jsonl"
        line = json.dumps(entry, default=str) + "\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()

    def save_result(self, result: OptimizationResult) -> None:
        """Save the final optimization result to result.json.

        Implements: REQ-ORCH-002
        Invariant: INV-ORCH-005 (atomic write)
        """
        path = self._workdir / "result.json"
        self._atomic_write(path, result.model_dump_json(indent=2))
