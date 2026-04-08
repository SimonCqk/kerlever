"""Spec Builder resolver — reference kernel resolution.

Resolves inline CUDA source, file:// URIs, and https:// URLs
into validated CUDA source code.

Spec: docs/spec-builder/spec.md §6.1
"""

from __future__ import annotations

import asyncio
import urllib.request
from pathlib import Path


async def resolve_reference_kernel(raw_value: str) -> str:
    """Resolve a reference_kernel value to inline CUDA source.

    Detects the form of the raw value and fetches/reads content accordingly:
    - ``file:///path`` reads a local file.
    - ``https://url`` fetches via HTTP GET.
    - Otherwise treats the value as inline CUDA source.

    After resolution the content is checked for ``__global__`` or
    ``__device__`` substrings to ensure it looks like CUDA.

    Args:
        raw_value: Raw reference_kernel field value from the spec YAML.

    Returns:
        Resolved CUDA source code string.

    Raises:
        FileNotFoundError: If a ``file:///`` path does not exist.
        ValueError: If an ``https://`` fetch fails or if the resolved
            content does not contain CUDA markers.

    Implements: REQ-SB-002, SCN-SB-002-01 through SCN-SB-002-05
    """
    if raw_value.startswith("file://"):
        content = _resolve_file(raw_value)
    elif raw_value.startswith("https://"):
        content = await _resolve_url(raw_value)
    else:
        content = raw_value

    _check_cuda_markers(content)
    return content


def _resolve_file(raw_value: str) -> str:
    """Read a local file referenced by a ``file://`` URI."""
    file_path = Path(raw_value.removeprefix("file://"))
    if not file_path.exists():
        raise FileNotFoundError(f"Reference kernel file not found: {file_path}")
    return file_path.read_text(encoding="utf-8")


async def _resolve_url(url: str) -> str:
    """Fetch content from an ``https://`` URL."""

    def _fetch() -> str:
        req = urllib.request.Request(url)  # noqa: S310
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            if resp.status < 200 or resp.status >= 300:
                raise ValueError(f"HTTP fetch failed for {url}: status {resp.status}")
            body: str = resp.read().decode("utf-8")
            return body

    try:
        return await asyncio.to_thread(_fetch)
    except urllib.error.HTTPError as exc:
        raise ValueError(f"HTTP fetch failed for {url}: status {exc.code}") from None
    except urllib.error.URLError as exc:
        raise ValueError(f"HTTP fetch failed for {url}: {exc.reason}") from None


def _check_cuda_markers(content: str) -> None:
    """Verify that the content contains CUDA function markers."""
    if "__global__" not in content and "__device__" not in content:
        raise ValueError(
            "Resolved reference kernel does not appear to be valid CUDA source: "
            "no __global__ or __device__ marker found"
        )
