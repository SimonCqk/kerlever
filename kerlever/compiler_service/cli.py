"""One-shot CLI: ``python -m kerlever.compiler_service``.

Useful for smoke-testing the service end-to-end without the HTTP layer.

Spec: docs/compiler-service/spec.md §6.12
Design: docs/compiler-service/design.md §13
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from kerlever.compiler_service.config import ServiceConfig
from kerlever.compiler_service.toolchain import ToolchainProbe
from kerlever.compiler_service.types import CompileRequest


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m kerlever.compiler_service``."""
    parser = argparse.ArgumentParser(
        prog="python -m kerlever.compiler_service",
        description="Compiler Service one-shot CLI",
    )
    parser.add_argument(
        "--request-json",
        type=Path,
        default=None,
        help="Path to a CompileRequest JSON file.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="If set, write the CompileResult JSON here.",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Run ToolchainProbe and exit.",
    )
    args = parser.parse_args(argv)

    config = ServiceConfig.from_env()

    if args.probe_only:
        probe = ToolchainProbe(config).run()
        payload = {
            "ok": probe.ok,
            "missing": probe.missing,
            "notes": probe.notes,
            "nvcc_version": probe.nvcc_version,
            "driver_version": probe.driver_version,
            "gpu_name": probe.gpu_name,
            "gpu_uuid": probe.gpu_uuid,
            "sanitizer_version": probe.sanitizer_version,
            "artifact_root_writable": probe.artifact_root_writable,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if probe.ok else 1

    if args.request_json is None:
        parser.error("--request-json is required unless --probe-only is set")

    request_path: Path = args.request_json
    result_json = asyncio.run(
        _run_request(config, request_path.read_text(encoding="utf-8"))
    )
    if args.output_json is not None:
        args.output_json.write_text(result_json)
    else:
        print(result_json)
    return 0


async def _run_request(config: ServiceConfig, body: str) -> str:
    """Build the same deps as ``create_app`` and compile one request."""
    # Deferred import: keeps the CLI path off the FastAPI extra.
    from kerlever.compiler_service.api.app import build_deps
    from kerlever.compiler_service.service import CompilerService

    request = CompileRequest.model_validate_json(body)
    deps = await build_deps(config)
    service = CompilerService(deps)
    result = await service.compile(request)
    return result.model_dump_json(indent=2)


if __name__ == "__main__":
    sys.exit(main())
