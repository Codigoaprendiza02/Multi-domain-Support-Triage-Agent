"""Shared ingestion data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""