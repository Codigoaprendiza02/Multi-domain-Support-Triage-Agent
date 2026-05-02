"""Shared retrieval data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ingestion.models import Chunk


@dataclass(slots=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    source: str = ""


@dataclass(slots=True)
class RetrievalResult:
    chunks: list[ScoredChunk] = field(default_factory=list)
    retrieved_companies: list[str] = field(default_factory=list)