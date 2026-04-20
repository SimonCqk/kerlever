"""Deterministic hashing helpers for the Compiler Service.

All functions are pure: no I/O, no state. They form the reproducibility
anchor for ``artifact_key`` (spec §6.1, REQ-CS-008).

Spec: docs/compiler-service/spec.md §6.1
Design: docs/compiler-service/design.md §4.6
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from pydantic import BaseModel

from kerlever.compiler_service.types import KernelExecutionSpec
from kerlever.types import ProblemSpec


def _sha256_hex(payload: bytes) -> str:
    """Return lower-case hex SHA-256 of ``payload``."""
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(model: BaseModel) -> bytes:
    """Pydantic-canonical JSON bytes for stable hashing.

    Sorted keys + ASCII-only guarantees identical bytes for the same
    logical content across Python invocations.
    """
    return model.model_dump_json().encode("utf-8")


def source_hash(source_code: str) -> str:
    """Return SHA-256 of ``source_code`` bytes.

    Implements: REQ-CS-008
    """
    return _sha256_hex(source_code.encode("utf-8"))


def problem_spec_hash(ps: ProblemSpec) -> str:
    """Return SHA-256 of the canonical-JSON serialization of ``ps``.

    Implements: REQ-CS-008
    """
    return _sha256_hex(_canonical_json_bytes(ps))


def launch_spec_hash(es: KernelExecutionSpec) -> str:
    """Return SHA-256 of the canonical-JSON serialization of ``es``.

    Implements: REQ-CS-008
    """
    return _sha256_hex(_canonical_json_bytes(es))


def compile_flags_hash(flags: Sequence[str]) -> str:
    """Return SHA-256 of the canonical serialization of ``flags``.

    Order matters; the same flags in a different order produce a different
    hash.
    """
    joined = "\n".join(flags)
    return _sha256_hex(joined.encode("utf-8"))


def toolchain_hash(info: Mapping[str, str]) -> str:
    """Return SHA-256 of a deterministic serialization of toolchain info.

    Only the keys ``nvcc_version``, ``driver_version``, ``gpu_uuid``,
    ``sanitizer_version`` are considered; unknown keys are ignored.
    """
    relevant = ("nvcc_version", "driver_version", "gpu_uuid", "sanitizer_version")
    parts = [f"{k}={info.get(k, '')}" for k in relevant]
    return _sha256_hex("|".join(parts).encode("utf-8"))


def artifact_key(
    *,
    source_hash: str,
    problem_spec_hash: str,
    launch_spec_hash: str,
    target_arch: str,
    toolchain_hash: str,
    compile_flags_hash: str,
    adapter_version: str,
    legacy_inferred_execution_spec: bool,
) -> str:
    """Return the deterministic ``artifact_key`` for a request.

    Any change in any of these inputs produces a different key.
    Implements: REQ-CS-008
    Invariant: INV-CS-009 (reuse asserts key equality)
    """
    flag = "1" if legacy_inferred_execution_spec else "0"
    parts = (
        source_hash,
        problem_spec_hash,
        launch_spec_hash,
        target_arch,
        toolchain_hash,
        compile_flags_hash,
        adapter_version,
        flag,
    )
    return _sha256_hex("|".join(parts).encode("utf-8"))
