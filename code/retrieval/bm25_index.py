"""Deterministic BM25-style keyword index."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from ingestion.models import Chunk

from .models import ScoredChunk


class BM25Index:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._document_frequencies: dict[str, int] = defaultdict(int)
        self._term_frequencies: list[Counter[str]] = []
        self._average_length = 0.0

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        self._term_frequencies = []
        self._document_frequencies = defaultdict(int)

        total_length = 0
        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            total_length += len(tokens)
            frequencies = Counter(tokens)
            self._term_frequencies.append(frequencies)
            for token in frequencies:
                self._document_frequencies[token] += 1

        self._average_length = total_length / len(chunks) if chunks else 0.0

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        if not self._chunks:
            return []

        query_tokens = self._tokenize(query)
        scored: list[ScoredChunk] = []
        for chunk, frequencies in zip(self._chunks, self._term_frequencies):
            score = self._score(query_tokens, frequencies, len(self._tokenize(chunk.text)))
            scored.append(ScoredChunk(chunk=chunk, score=score, source="bm25"))

        scored.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return scored[:top_k]

    def _score(self, query_tokens: list[str], frequencies: Counter[str], document_length: int) -> float:
        if not query_tokens or document_length == 0:
            return 0.0

        k1 = 1.5
        b = 0.75
        score = 0.0
        for token in query_tokens:
            tf = frequencies.get(token, 0)
            if not tf:
                continue
            df = self._document_frequencies.get(token, 0)
            idf = math.log(1 + ((len(self._chunks) - df + 0.5) / (df + 0.5)))
            denominator = tf + k1 * (1 - b + b * (document_length / self._average_length if self._average_length else 1.0))
            score += idf * ((tf * (k1 + 1)) / denominator)
        return score

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