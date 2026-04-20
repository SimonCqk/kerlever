"""FastAPI surface for the Compiler Service.

Spec: docs/compiler-service/spec.md §6.12
Design: docs/compiler-service/design.md §12
"""

from __future__ import annotations

from kerlever.compiler_service.api.app import create_app

__all__ = ["create_app"]
