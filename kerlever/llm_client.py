"""Kerlever LLM client — shared LLM client protocol and Anthropic implementation.

Extracted from spec_builder.llm_judge to enable reuse by both the
Spec Builder and Strategy Navigator modules.

Spec: docs/navigator/spec.md §5 (LLM Client Protocol)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Protocol for LLM client injection.

    Enables stub/mock injection for testing while the production
    implementation uses the Anthropic Python SDK.
    """

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt to the LLM and return the text response."""
        ...


class AnthropicLLMClient:
    """Production LLM client using the Anthropic Python SDK.

    Wraps ``anthropic.AsyncAnthropic`` to satisfy ``LLMClientProtocol``.
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514") -> None:
        import anthropic

        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt to Claude and return the text response."""
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text  # type: ignore[union-attr]
