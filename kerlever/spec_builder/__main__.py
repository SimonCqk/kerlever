"""Spec Builder CLI — run via ``python -m kerlever.spec_builder``.

Supports ``--validate`` (batch) and ``--interactive`` modes.

Spec: docs/spec-builder/spec.md §6.6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from kerlever.llm_client import AnthropicLLMClient, LLMClientProtocol
from kerlever.spec_builder import validate_spec
from kerlever.spec_builder.interactive import interactive_collect
from kerlever.spec_builder.types import ValidationResult
from kerlever.types import ProblemSpec


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m kerlever.spec_builder",
        description=(
            "Kerlever Spec Builder: validate or interactively build a ProblemSpec."
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--validate",
        metavar="PATH",
        type=str,
        help="Validate a YAML spec file",
    )
    group.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive field collection",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip Stage 2 LLM judge (no API key needed for --validate)",
    )
    return parser


def _create_llm_client() -> LLMClientProtocol:
    """Create an Anthropic LLM client from the environment.

    Returns:
        An AnthropicLLMClient configured with the ANTHROPIC_API_KEY.

    Raises:
        SystemExit: If no API key is available.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print(
            "Error: ANTHROPIC_API_KEY environment variable is not set. "
            "Use --no-llm to skip LLM validation.",
            file=sys.stderr,
        )
        sys.exit(1)
    return AnthropicLLMClient(api_key=api_key)


async def _run_validate(path_str: str, no_llm: bool) -> None:
    """Run batch validation on a YAML file."""
    path = Path(path_str)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        raw = path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
    except (OSError, yaml.YAMLError) as exc:
        print(f"Error: failed to load YAML: {exc}", file=sys.stderr)
        sys.exit(1)

    # Pydantic construction — convert validation errors to schema issues
    try:
        spec = ProblemSpec.model_validate(data)
    except ValidationError as exc:
        from kerlever.spec_builder.types import ValidationIssue

        issues = [
            ValidationIssue(
                dimension="schema",
                severity="fail",
                message=str(err),
            )
            for err in exc.errors()
        ]
        result = ValidationResult(issues=issues)
        print(json.dumps(result.model_dump(), indent=2))
        sys.exit(1)

    # Build LLM client if needed
    llm_client: LLMClientProtocol | None = None
    if not no_llm:
        llm_client = _create_llm_client()

    result = await validate_spec(spec, llm_client=llm_client)
    print(json.dumps(result.model_dump(), indent=2))

    if result.is_valid:
        sys.exit(0)
    else:
        sys.exit(1)


async def _run_interactive(no_llm: bool) -> None:
    """Run interactive collection mode."""
    llm_client = _create_llm_client()
    await interactive_collect(llm_client, no_llm=no_llm)


def main() -> None:
    """CLI entry point for the Spec Builder module."""
    parser = _build_parser()
    args = parser.parse_args()

    if not args.validate and not args.interactive:
        parser.print_usage(sys.stderr)
        sys.exit(2)

    if args.validate:
        asyncio.run(_run_validate(args.validate, args.no_llm))
    elif args.interactive:
        asyncio.run(_run_interactive(args.no_llm))


if __name__ == "__main__":
    main()
