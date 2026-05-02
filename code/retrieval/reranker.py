"""Simple deterministic reranker."""

from __future__ import annotations

from .models import ScoredChunk


class Reranker:
    def rerank(self, query: str, chunks: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        query_tokens = self._tokenize(query)
        reranked = []
        for item in chunks:
            chunk_tokens = self._tokenize(item.chunk.text)
            overlap = len(set(query_tokens) & set(chunk_tokens))
            score = item.score + overlap * 0.01
            reranked.append(ScoredChunk(chunk=item.chunk, score=score, source=item.source or "reranker"))

        reranked.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return reranked[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        return [token.lower().strip(".,!?;:\"'()[]{}") for token in text.split() if token.strip()]