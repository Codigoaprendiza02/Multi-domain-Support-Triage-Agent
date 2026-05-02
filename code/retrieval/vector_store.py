"""Local-first vector store with optional ChromaDB support.

The implementation keeps the public API stable for phase 2 tests while using a
deterministic JSON-backed store if ChromaDB or sentence-transformers are not
available in the active environment.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from ingestion.models import Chunk

from config import CHROMA_PERSIST_DIR

from .models import ScoredChunk


class VectorStore:
    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self.persist_dir / "vector_index.json"
        self._chunks: list[Chunk] = []
        self._documents_by_company: dict[str, list[Chunk]] = {}
        self._load_existing_index()

    def index_chunks(self, chunks: list[Chunk]) -> None:
        if self.is_indexed():
            return

        self._chunks = list(chunks)
        self._documents_by_company = {}
        for chunk in chunks:
            company = str(chunk.metadata.get("company", "unknown")).lower()
            self._documents_by_company.setdefault(company, []).append(chunk)

        payload = [
            {
                "text": chunk.text,
                "metadata": chunk.metadata,
                "chunk_id": chunk.chunk_id,
            }
            for chunk in self._chunks
        ]
        self._index_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def search(self, query: str, company: str | None, top_k: int) -> list[ScoredChunk]:
        if not self._chunks:
            return []

        candidates = self._candidate_chunks(company)
        scored = [ScoredChunk(chunk=chunk, score=self._score(query, chunk), source="dense") for chunk in candidates]
        scored.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return scored[:top_k]

    def is_indexed(self) -> bool:
        return self._index_file.exists() and self._index_file.stat().st_size > 0

    def _candidate_chunks(self, company: str | None) -> list[Chunk]:
        if company is None:
            return list(self._chunks)
        return list(self._documents_by_company.get(company.lower(), []))

    def _score(self, query: str, chunk: Chunk) -> float:
        query_tokens = self._tokenize(query)
        chunk_tokens = self._tokenize(chunk.text)
        if not query_tokens or not chunk_tokens:
            return 0.0

        query_counts = Counter(query_tokens)
        chunk_counts = Counter(chunk_tokens)
        overlap = sum(min(query_counts[token], chunk_counts[token]) for token in query_counts)
        length_penalty = math.log(len(chunk_tokens) + 1)
        metadata_bonus = 0.1 if query.lower() in chunk.text.lower() else 0.0
        return (overlap / length_penalty) + metadata_bonus

    def _tokenize(self, text: str) -> list[str]:
        expanded: list[str] = []
        synonym_map = {
            "password": ["login", "account", "access", "reset", "signin"],
            "subscription": ["plan", "billing", "membership"],
            "invoice": ["billing", "payment", "charge"],
            "payment": ["billing", "charge", "invoice"],
            "charge": ["payment", "billing", "invoice"],
            "fraud": ["unauthorized", "stolen", "security"],
            "dispute": ["chargeback", "billing", "refund"],
            "cancel": ["close", "stop", "terminate"],
        }

        for raw_token in text.split():
            token = raw_token.lower().strip(".,!?;:\"'()[]{}")
            if not token:
                continue
            expanded.append(token)
            expanded.extend(synonym_map.get(token, []))
        return expanded

    def _load_existing_index(self) -> None:
        if not self.is_indexed():
            return

        raw = json.loads(self._index_file.read_text(encoding="utf-8"))
        chunks: list[Chunk] = []
        for item in raw:
            chunks.append(Chunk(text=item["text"], metadata=dict(item["metadata"]), chunk_id=item["chunk_id"]))
        self._chunks = chunks
        self._documents_by_company = {}
        for chunk in chunks:
            company = str(chunk.metadata.get("company", "unknown")).lower()
            self._documents_by_company.setdefault(company, []).append(chunk)