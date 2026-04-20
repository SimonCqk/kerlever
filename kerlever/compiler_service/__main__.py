"""Entry point: ``python -m kerlever.compiler_service``.

Spec: docs/compiler-service/spec.md §6.12
Design: docs/compiler-service/design.md §13
"""

from __future__ import annotations

import sys

from kerlever.compiler_service.cli import main

if __name__ == "__main__":
    sys.exit(main())
