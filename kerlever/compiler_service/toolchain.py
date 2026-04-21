"""Toolchain probe + external-tool subprocess wrappers.

The module deliberately does NOT import ``cuda-python`` at module load:
the package must be importable on CPU-only hosts (see assignment notes).
Driver-API access is behind ``DriverApiAttributes.try_load``, which does a
guarded import and returns ``None`` on failure.

Spec: docs/compiler-service/spec.md §6.1, §6.4
Design: docs/compiler-service/design.md §8, §14
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from kerlever.compiler_service.config import ServiceConfig
from kerlever.compiler_service.identity import toolchain_hash
from kerlever.compiler_service.types import ToolchainInfo

logger = logging.getLogger(__name__)


_PTXAS_REGISTERS_RE = re.compile(r"Used\s+(\d+)\s+registers", re.IGNORECASE)
_PTXAS_SMEM_RE = re.compile(r"(\d+)\s+bytes\s+smem", re.IGNORECASE)
_PTXAS_STACK_SMEM_RE = re.compile(
    r"(\d+)\s+bytes\s+(?:stack\s+frame|shared\s+memory)", re.IGNORECASE
)
_PTXAS_SPILL_STORES_RE = re.compile(r"(\d+)\s+bytes\s+spill\s+stores", re.IGNORECASE)
_PTXAS_SPILL_LOADS_RE = re.compile(r"(\d+)\s+bytes\s+spill\s+loads", re.IGNORECASE)


@dataclass(frozen=True)
class PtxasReport:
    """Parsed ``-Xptxas=-v`` output (spec §6.5 source preference)."""

    registers_per_thread: int | None = None
    smem_bytes_per_block: int | None = None
    spill_stores: int | None = None
    spill_loads: int | None = None


class PtxasParser:
    """Pure parser for ``ptxas`` verbose output (no subprocess).

    Consumes the combined stdout+stderr of an ``nvcc -Xptxas=-v`` run;
    returns ``None`` for any unreadable fact (INV-CS-003).
    """

    @staticmethod
    def parse(stdout: str, stderr: str) -> PtxasReport:
        """Extract ptxas facts from the combined compile output."""
        combined = f"{stdout}\n{stderr}"

        registers = _match_int(_PTXAS_REGISTERS_RE, combined)
        smem = _match_int(_PTXAS_SMEM_RE, combined)
        spill_stores = _match_int(_PTXAS_SPILL_STORES_RE, combined)
        spill_loads = _match_int(_PTXAS_SPILL_LOADS_RE, combined)

        return PtxasReport(
            registers_per_thread=registers,
            smem_bytes_per_block=smem,
            spill_stores=spill_stores,
            spill_loads=spill_loads,
        )


@dataclass(frozen=True)
class NvccResult:
    """Structured outcome of one nvcc invocation.

    Design §8.1. Byte-bounded stdout/stderr; ``truncated`` tells the caller
    whether a truncation marker should be appended to the excerpt.
    """

    returncode: int
    stdout_excerpt: str
    stderr_excerpt: str
    truncated: bool
    command: str
    timed_out: bool


class NvccRunner:
    """Async subprocess wrapper around ``nvcc``.

    All flags are driven by ``ServiceConfig.default_compile_flags`` — the
    runner never composes flags ad-hoc (spec §6.4).
    """

    def __init__(self, nvcc_path: Path, config: ServiceConfig) -> None:
        self._nvcc_path = nvcc_path
        self._config = config

    async def compile(
        self,
        source: Path,
        output: Path,
        target_arch: str,
        extra_flags: Sequence[str] = (),
        timeout_s: float | None = None,
        max_log_bytes: int | None = None,
    ) -> NvccResult:
        """Compile ``source`` → ``output`` and return the structured result.

        Implements: REQ-CS-001, REQ-CS-002
        """
        flags = [
            *self._config.default_compile_flags,
            f"-arch={target_arch}",
            *extra_flags,
            "-o",
            str(output),
            str(source),
        ]
        timeout = timeout_s or self._config.compile_timeout_s
        byte_cap = max_log_bytes or self._config.max_log_bytes
        argv = [str(self._nvcc_path), *flags]
        command = " ".join(argv)

        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"nvcc not found at {self._nvcc_path}") from exc

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except TimeoutError:
            process.kill()
            with contextlib.suppress(Exception):
                await process.wait()
            return NvccResult(
                returncode=-1,
                stdout_excerpt="",
                stderr_excerpt="",
                truncated=False,
                command=command,
                timed_out=True,
            )

        stdout_text, stdout_trunc = _bound_bytes(stdout_bytes, byte_cap)
        stderr_text, stderr_trunc = _bound_bytes(stderr_bytes, byte_cap)

        return NvccResult(
            returncode=process.returncode or 0,
            stdout_excerpt=stdout_text,
            stderr_excerpt=stderr_text,
            truncated=stdout_trunc or stderr_trunc,
            command=command,
            timed_out=False,
        )


@dataclass(frozen=True)
class CuobjdumpResult:
    """Structured outcome of a ``cuobjdump`` invocation.

    ``output_path`` is the path to the written artifact (SASS, PTX, or
    cubin depending on which ``CuobjdumpRunner`` method produced it), or
    ``None`` when the tool produced no usable output (INV-CS-003: missing
    → None, never fabricated).
    """

    returncode: int
    output_path: Path | None
    stderr_excerpt: str
    timed_out: bool


class CuobjdumpRunner:
    """Wrapper around ``cuobjdump --dump-sass`` / ``--extract-elf`` / ``--dump-ptx``."""

    def __init__(self, cuobjdump_path: Path, config: ServiceConfig) -> None:
        self._cuobjdump_path = cuobjdump_path
        self._config = config

    async def extract_sass(
        self, executable: Path, output: Path, timeout_s: float | None = None
    ) -> CuobjdumpResult:
        """Dump SASS of ``executable`` to ``output``."""
        argv = [str(self._cuobjdump_path), "--dump-sass", str(executable)]
        timeout = timeout_s or self._config.cuobjdump_timeout_s
        stdout_bytes, stderr_bytes, returncode, timed_out = await self._run_argv(
            argv, timeout
        )
        stderr_text, _ = _bound_bytes(stderr_bytes, self._config.max_log_bytes)
        if timed_out:
            return CuobjdumpResult(
                returncode=-1,
                output_path=None,
                stderr_excerpt=stderr_text,
                timed_out=True,
            )
        if returncode == 0 and stdout_bytes:
            # Bound the SASS size to avoid runaway disk usage.
            capped = stdout_bytes[: self._config.max_artifact_bytes]
            output.write_bytes(capped)
            return CuobjdumpResult(
                returncode=0,
                output_path=output,
                stderr_excerpt=stderr_text,
                timed_out=False,
            )
        return CuobjdumpResult(
            returncode=returncode,
            output_path=None,
            stderr_excerpt=stderr_text,
            timed_out=False,
        )

    async def dump_ptx(
        self,
        executable: Path,
        output: Path,
        timeout_s: float | None = None,
    ) -> CuobjdumpResult:
        """Dump PTX of ``executable`` to ``output`` (best-effort).

        Runs ``cuobjdump --dump-ptx <exec>`` and captures stdout. Missing
        PTX (e.g. cubin-only fat binaries) is reported as
        ``output_path=None`` — callers must treat it as non-fatal.
        """
        argv = [str(self._cuobjdump_path), "--dump-ptx", str(executable)]
        timeout = timeout_s or self._config.cuobjdump_timeout_s
        stdout_bytes, stderr_bytes, returncode, timed_out = await self._run_argv(
            argv, timeout
        )
        stderr_text, _ = _bound_bytes(stderr_bytes, self._config.max_log_bytes)
        if timed_out:
            return CuobjdumpResult(
                returncode=-1,
                output_path=None,
                stderr_excerpt=stderr_text,
                timed_out=True,
            )
        if returncode == 0 and stdout_bytes:
            capped = stdout_bytes[: self._config.max_artifact_bytes]
            output.write_bytes(capped)
            return CuobjdumpResult(
                returncode=0,
                output_path=output,
                stderr_excerpt=stderr_text,
                timed_out=False,
            )
        return CuobjdumpResult(
            returncode=returncode,
            output_path=None,
            stderr_excerpt=stderr_text,
            timed_out=False,
        )

    async def extract_cubin(
        self,
        executable: Path,
        output: Path,
        target_arch: str,
        timeout_s: float | None = None,
    ) -> CuobjdumpResult:
        """Extract an ELF cubin matching ``target_arch`` from ``executable``.

        Runs ``cuobjdump --extract-elf all <executable>`` into a tmp dir,
        then picks the emitted ``.cubin`` whose filename encodes the
        requested ``target_arch`` (e.g. ``sm_80``). If multiple cubins
        match, the first is kept. Missing cubin is reported as
        ``output_path=None`` (INV-CS-003: missing = None, never
        fabricated).
        """
        import tempfile

        timeout = timeout_s or self._config.cuobjdump_timeout_s
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            argv = [
                str(self._cuobjdump_path),
                "--extract-elf",
                "all",
                str(executable),
            ]
            stdout_bytes, stderr_bytes, returncode, timed_out = await self._run_argv(
                argv, timeout, cwd=tmpdir
            )
            del stdout_bytes  # command only writes to cwd, not stdout
            stderr_text, _ = _bound_bytes(stderr_bytes, self._config.max_log_bytes)
            if timed_out:
                return CuobjdumpResult(
                    returncode=-1,
                    output_path=None,
                    stderr_excerpt=stderr_text,
                    timed_out=True,
                )
            if returncode != 0:
                return CuobjdumpResult(
                    returncode=returncode,
                    output_path=None,
                    stderr_excerpt=stderr_text,
                    timed_out=False,
                )

            match = _pick_cubin_for_arch(tmpdir, target_arch)
            if match is None:
                return CuobjdumpResult(
                    returncode=0,
                    output_path=None,
                    stderr_excerpt=stderr_text,
                    timed_out=False,
                )
            data = match.read_bytes()[: self._config.max_artifact_bytes]
            output.write_bytes(data)
            return CuobjdumpResult(
                returncode=0,
                output_path=output,
                stderr_excerpt=stderr_text,
                timed_out=False,
            )

    async def _run_argv(
        self,
        argv: list[str],
        timeout: float,
        cwd: Path | None = None,
    ) -> tuple[bytes, bytes, int, bool]:
        """Run ``argv`` and return ``(stdout, stderr, returncode, timed_out)``."""
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd) if cwd is not None else None,
            )
        except FileNotFoundError:
            err = f"cuobjdump not found at {self._cuobjdump_path}".encode()
            return (b"", err, -1, False)
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except TimeoutError:
            process.kill()
            with contextlib.suppress(Exception):
                await process.wait()
            return (b"", b"", -1, True)
        return (stdout_bytes, stderr_bytes, process.returncode or 0, False)


class DriverApiAttributes:
    """Thin facade over ``cuda-python`` ``cuFuncGetAttribute``.

    The module-level import of ``cuda`` is DEFERRED to ``try_load`` so the
    package is importable on CPU-only hosts (INV-CS-003 fallback path).
    """

    def __init__(self, driver_module: object) -> None:
        self._driver = driver_module

    @classmethod
    def try_load(cls) -> DriverApiAttributes | None:
        """Attempt to import ``cuda-python``; return ``None`` on failure.

        The caller (``StaticResourceExtractor``) falls back to ``ptxas`` when
        this returns ``None`` — never fabricating a value.
        """
        try:
            import importlib

            driver = importlib.import_module("cuda.bindings.driver")
        except ImportError:
            return None
        except Exception as exc:  # noqa: BLE001 — broad on-purpose
            logger.warning("cuda_python_import_failed", extra={"error": str(exc)})
            return None
        return cls(driver)

    def read_registers_per_thread(self, binary: Path, entrypoint: str) -> int | None:
        """Read ``CU_FUNC_ATTRIBUTE_NUM_REGS`` for ``entrypoint``.

        Returns ``None`` on any lookup error — the caller falls back to ptxas
        without fabricating data.
        """
        return self._read_attribute(binary, entrypoint, "CU_FUNC_ATTRIBUTE_NUM_REGS")

    def read_static_smem_bytes(self, binary: Path, entrypoint: str) -> int | None:
        """Read ``CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES`` for ``entrypoint``."""
        return self._read_attribute(
            binary, entrypoint, "CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES"
        )

    def read_max_threads_per_block(self, binary: Path, entrypoint: str) -> int | None:
        """Read ``CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK`` for ``entrypoint``."""
        return self._read_attribute(
            binary, entrypoint, "CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK"
        )

    def _read_attribute(
        self, binary: Path, entrypoint: str, attribute_name: str
    ) -> int | None:
        """Load a cubin/PTX module and read one function attribute.

        The cuda-python bindings have changed package names over time; this
        facade keeps the interaction defensive. Any import, load, symbol, or
        enum mismatch returns ``None`` and lets the ptxas fallback stand.
        """
        if not binary.exists() or not entrypoint:
            return None
        driver = self._driver
        module = None
        try:
            init = getattr(driver, "cuInit", None)
            if init is not None and not _cuda_ok(init(0)):
                return None

            load_data = getattr(driver, "cuModuleLoadData", None)
            get_function = getattr(driver, "cuModuleGetFunction", None)
            get_attribute = getattr(driver, "cuFuncGetAttribute", None)
            unload = getattr(driver, "cuModuleUnload", None)
            if load_data is None or get_function is None or get_attribute is None:
                return None

            loaded = load_data(binary.read_bytes())
            if not _cuda_ok(loaded):
                return None
            module = _cuda_payload(loaded)
            if module is None:
                return None

            func_result = get_function(module, entrypoint.encode("utf-8"))
            if not _cuda_ok(func_result):
                return None
            function = _cuda_payload(func_result)
            if function is None:
                return None

            attribute = _cuda_enum_value(driver, attribute_name)
            if attribute is None:
                return None
            attr_result = get_attribute(attribute, function)
            if not _cuda_ok(attr_result):
                return None
            value = _cuda_payload(attr_result)
            return value if isinstance(value, int) else None
        except Exception as exc:  # noqa: BLE001 — best-effort fallback
            logger.warning(
                "cuda_driver_attribute_read_failed",
                extra={"attribute": attribute_name, "error": str(exc)},
            )
            return None
        finally:
            if module is not None:
                unload = getattr(driver, "cuModuleUnload", None)
                if unload is not None:
                    with contextlib.suppress(Exception):
                        unload(module)


def _cuda_ok(result: object) -> bool:
    """Return True when a cuda-python call result reports success."""
    if isinstance(result, (tuple, list)):
        if not result:
            return False
        status = result[0]
    else:
        status = result
    value = getattr(status, "value", status)
    name = getattr(status, "name", "")
    return value == 0 or name == "CUDA_SUCCESS"


def _cuda_payload(result: object) -> object | None:
    """Return the first payload value from a cuda-python tuple result."""
    if isinstance(result, (tuple, list)) and len(result) >= 2:
        return cast(object, result[1])
    return None


def _cuda_enum_value(driver: object, attribute_name: str) -> object | None:
    """Resolve a CUfunction_attribute enum value across binding versions."""
    for container_name in (
        "CUfunction_attribute",
        "CUfunction_attribute_enum",
    ):
        container = getattr(driver, container_name, None)
        if container is not None and hasattr(container, attribute_name):
            return cast(object, getattr(container, attribute_name))
    return cast(object | None, getattr(driver, attribute_name, None))


@dataclass(frozen=True)
class ToolchainProbeResult:
    """Outcome of the startup probe.

    Used both by ``create_app`` (for ``SystemExit(1)``) and by
    ``GET /healthz`` (for 503 body) — INV-CS-012 requires ONE function and
    TWO call sites.
    """

    ok: bool
    nvcc_path: Path | None
    cuobjdump_path: Path | None
    sanitizer_path: Path | None
    nvcc_version: str | None
    driver_version: str | None
    gpu_name: str | None
    gpu_uuid: str | None
    sanitizer_version: str | None
    artifact_root_writable: bool
    cuda_python_available: bool
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_error_json(self) -> str:
        """Serialize a human-debuggable error body for ``sys.stderr``."""
        import json

        payload = {
            "ok": self.ok,
            "missing": self.missing,
            "notes": self.notes,
            "nvcc_version": self.nvcc_version,
            "driver_version": self.driver_version,
            "gpu_name": self.gpu_name,
            "gpu_uuid": self.gpu_uuid,
            "sanitizer_version": self.sanitizer_version,
            "artifact_root_writable": self.artifact_root_writable,
        }
        return json.dumps(payload, sort_keys=True)


class ToolchainProbe:
    """Checks nvcc/driver/GPU/sanitizer/artifact root (spec §6.12 /healthz).

    The same method is called from the FastAPI lifespan (which exits
    non-zero on failure) and from the ``/healthz`` handler — INV-CS-012.
    """

    def __init__(self, config: ServiceConfig) -> None:
        self._config = config

    def run(self) -> ToolchainProbeResult:
        """Run every startup check and return the aggregate result."""
        missing: list[str] = []
        notes: list[str] = []

        nvcc_path = _resolve_tool(self._config.nvcc_path, "nvcc")
        cuobjdump_path = _resolve_tool(self._config.cuobjdump_path, "cuobjdump")
        sanitizer_path = _resolve_tool(self._config.sanitizer_path, "compute-sanitizer")
        nvcc_version = (
            _run_version([str(nvcc_path), "--version"]) if nvcc_path else None
        )
        if nvcc_path is None or nvcc_version is None:
            missing.append("nvcc")

        if cuobjdump_path is None:
            missing.append("cuobjdump")

        sanitizer_version = (
            _run_version([str(sanitizer_path), "--version"]) if sanitizer_path else None
        )
        if sanitizer_path is None or sanitizer_version is None:
            missing.append("compute-sanitizer")

        smi_path = _resolve_tool(self._config.nvidia_smi_path, "nvidia-smi")
        driver_version, gpu_name, gpu_uuid = _probe_nvidia_smi(smi_path)
        if smi_path is None or driver_version is None:
            missing.append("nvidia-smi/driver")

        artifact_writable = _ensure_writable(self._config.kerlever_artifact_root)
        if not artifact_writable:
            missing.append("artifact_root")

        cuda_python_available = DriverApiAttributes.try_load() is not None
        if not cuda_python_available:
            notes.append("cuda-python unavailable — ptxas fallback in use (INV-CS-003)")

        ok = not missing
        return ToolchainProbeResult(
            ok=ok,
            nvcc_path=nvcc_path,
            cuobjdump_path=cuobjdump_path,
            sanitizer_path=sanitizer_path,
            nvcc_version=nvcc_version,
            driver_version=driver_version,
            gpu_name=gpu_name,
            gpu_uuid=gpu_uuid,
            sanitizer_version=sanitizer_version,
            artifact_root_writable=artifact_writable,
            cuda_python_available=cuda_python_available,
            missing=missing,
            notes=notes,
        )

    def snapshot(self, probe_result: ToolchainProbeResult) -> ToolchainInfo:
        """Return a ``ToolchainInfo`` snapshot from a successful probe result.

        Raises ``ValueError`` when called on a failed probe — the caller
        is expected to have exited non-zero before ever reaching this.
        """
        if not probe_result.ok:
            raise ValueError("Cannot snapshot toolchain from a failed probe")
        info_map = {
            "nvcc_version": probe_result.nvcc_version or "",
            "driver_version": probe_result.driver_version or "",
            "gpu_uuid": probe_result.gpu_uuid or "",
            "sanitizer_version": probe_result.sanitizer_version or "",
        }
        return ToolchainInfo(
            nvcc_version=probe_result.nvcc_version or "",
            driver_version=probe_result.driver_version or "",
            gpu_name=probe_result.gpu_name or "",
            gpu_uuid=probe_result.gpu_uuid or "",
            sanitizer_version=probe_result.sanitizer_version or "",
            toolchain_hash=toolchain_hash(info_map),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_tool(configured: Path, name: str) -> Path | None:
    """Return the tool path if it exists and is executable, else None."""
    if configured.exists() and os.access(configured, os.X_OK):
        return configured
    found = shutil.which(name)
    if found:
        return Path(found)
    return None


def _run_version(argv: list[str]) -> str | None:
    """Run a ``--version`` command synchronously and return its stdout."""
    import subprocess

    try:
        completed = subprocess.run(  # noqa: S603 — argv fully controlled
            argv,
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        # nvcc emits --version to stdout; some tools use stderr.
        out = (completed.stdout or completed.stderr).strip()
        return out or None
    return (completed.stdout or completed.stderr).strip() or None


def _probe_nvidia_smi(
    smi_path: Path | None,
) -> tuple[str | None, str | None, str | None]:
    """Return ``(driver_version, gpu_name, gpu_uuid)`` from ``nvidia-smi``."""
    if smi_path is None:
        return (None, None, None)
    import subprocess

    argv = [
        str(smi_path),
        "--query-gpu=driver_version,name,uuid",
        "--format=csv,noheader",
    ]
    try:
        completed = subprocess.run(  # noqa: S603 — argv fully controlled
            argv,
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return (None, None, None)
    if completed.returncode != 0 or not completed.stdout.strip():
        return (None, None, None)
    line = completed.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        return (None, None, None)
    return (parts[0], parts[1], parts[2])


def _ensure_writable(root: Path) -> bool:
    """Create ``root`` (if missing) and verify a test-write succeeds."""
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".kerlever-probe"
        probe.write_text("ok")
        probe.unlink()
    except OSError:
        return False
    return True


def _bound_bytes(data: bytes, max_bytes: int) -> tuple[str, bool]:
    """Decode ``data`` as UTF-8, truncating to ``max_bytes`` with a marker."""
    if len(data) <= max_bytes:
        return (data.decode("utf-8", errors="replace"), False)
    head = data[:max_bytes].decode("utf-8", errors="replace")
    return (head + "\n[truncated]\n", True)


def _match_int(pattern: re.Pattern[str], text: str) -> int | None:
    """Return the first integer group match, or None."""
    match = pattern.search(text)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except (ValueError, IndexError):
        return None


def _pick_cubin_for_arch(tmpdir: Path, target_arch: str) -> Path | None:
    """Pick a cubin matching ``target_arch`` from ``cuobjdump --extract-elf``.

    ``cuobjdump --extract-elf all`` names outputs like
    ``<stem>.sm_80.cubin`` (one per compiled arch in the fat binary).
    This helper prefers a name containing the requested ``target_arch``
    literal; if none match it returns the first cubin it finds (so the
    caller gets a valid ELF even on unusual builds). Returns ``None`` if
    no ``.cubin`` files are present — the caller records
    ``cubin_artifact_id=None`` in that case (INV-CS-003).
    """
    cubins = sorted(tmpdir.glob("*.cubin"))
    if not cubins:
        return None
    for cubin in cubins:
        if target_arch in cubin.name:
            return cubin
    return cubins[0]
