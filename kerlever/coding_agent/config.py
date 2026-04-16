"""Coding Agent configuration — generation parameters.

Spec: docs/coding-agent/spec.md §5 (CodingAgentConfig)
"""

from __future__ import annotations

from pydantic import BaseModel


class CodingAgentConfig(BaseModel, frozen=True):
    """Configuration parameters for the Coding Agent.

    All fields have defaults. Passing None at construction uses all defaults.
    """

    max_code_length: int = 4096
    """Max characters in generated code block."""

    max_retries: int = 1
    """Retries per candidate on failure."""

    temperature_base: float = 0.7
    """Base LLM temperature (if supported)."""

    temperature_spread: float = 0.1
    """Variation between candidates."""
