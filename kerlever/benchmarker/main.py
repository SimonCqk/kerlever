"""Benchmarker — uvicorn entrypoint.

Exposes the module-level ``app`` symbol so ``uvicorn kerlever.benchmarker.main:app``
works; the ``__main__`` block lets you launch with
``python -m kerlever.benchmarker.main``.

NVML and cuda-python are **not** imported at module scope here. The
FastAPI lifespan hook performs NVML init inside :func:`create_app` so a
pure import succeeds on non-GPU machines (spec §7.5).

Spec: docs/benchmarker/spec.md §SC-BENCH-010
Design: docs/benchmarker/design.md §2.1 main.py, §8.1
"""

from __future__ import annotations

import argparse
import logging
import sys

from kerlever.benchmarker.config import BenchmarkerConfig
from kerlever.benchmarker.service import create_app


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build a minimal argv parser for the ``python -m`` entrypoint."""
    parser = argparse.ArgumentParser(prog="kerlever.benchmarker.main")
    parser.add_argument(
        "--host", default=None, help="Override KERLEVER_BENCH_BIND_HOST."
    )
    parser.add_argument("--port", type=int, default=None, help="Override port.")
    parser.add_argument(
        "--log-level", default=None, help="Override log level."
    )
    return parser


# Build the app at import time so ``uvicorn kerlever.benchmarker.main:app``
# can resolve the symbol without executing a separate factory function.
app = create_app()


def main(argv: list[str] | None = None) -> int:
    """Launch uvicorn with the FastAPI app.

    Args:
        argv: Optional argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        0 on clean shutdown, a non-zero integer on startup failure.
    """
    import uvicorn  # noqa: PLC0415 — lazy so tests can import main cheaply

    args = _build_arg_parser().parse_args(argv)
    cfg = BenchmarkerConfig.from_env()
    host = args.host or cfg.bind_host
    port = args.port or cfg.bind_port
    log_level = (args.log_level or cfg.log_level).lower()
    logging.basicConfig(level=log_level.upper())
    uvicorn.run(
        "kerlever.benchmarker.main:app",
        host=host,
        port=port,
        workers=1,
        log_level=log_level,
        lifespan="on",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["app", "main"]
