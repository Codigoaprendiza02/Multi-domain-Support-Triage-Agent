"""Combine dense and sparse retrieval results."""

from __future__ import annotations

from collections import defaultdict

from config import RERANK_TOP_K, TOP_K_RETRIEVAL

from .bm25_index import BM25Index
from .models import ScoredChunk
from .reranker import Reranker
from .vector_store import VectorStore


class HybridRetriever:
    def __init__(self, vector_store: VectorStore | None = None, bm25_index: BM25Index | None = None, reranker: Reranker | None = None) -> None:
        self.vector_store = vector_store or VectorStore()
        self.bm25_index = bm25_index or BM25Index()
        self.reranker = reranker or Reranker()
        self.dense_weight = 0.6
        self.sparse_weight = 0.4

    def retrieve(self, query: str, company: str | None, top_k: int) -> list[ScoredChunk]:
        candidate_pool = max(TOP_K_RETRIEVAL * 5, top_k * 5)
        dense_results = self.vector_store.search(query, company, top_k=candidate_pool)
        sparse_results = self.bm25_index.search(query, top_k=candidate_pool)

        combined = self._combine(dense_results, sparse_results)
        reranked = self.reranker.rerank(query, combined, top_k=len(combined))

        if company is not None:
            # Filter reranked results to the requested company only (case-insensitive)
            wanted = str(company).lower()
            filtered = [r for r in reranked if str(r.chunk.metadata.get("company", "")).lower() == wanted]
            return filtered[:top_k]

        if not combined:
            return []

        head = [dense_results[0]] if dense_results else ([combined[0]] if combined else [])
        diversified_tail = self._diversify_by_company(reranked[1:], max(0, top_k - 1))
        merged = head + diversified_tail
        deduped: list[ScoredChunk] = []
        seen_ids: set[str] = set()
        for item in merged:
            if item.chunk.chunk_id in seen_ids:
                continue
            deduped.append(item)
            seen_ids.add(item.chunk.chunk_id)
            if len(deduped) >= top_k:
                break
        return deduped

    def _combine(self, dense_results: list[ScoredChunk], sparse_results: list[ScoredChunk]) -> list[ScoredChunk]:
        combined_scores: dict[str, float] = defaultdict(float)
        chunk_by_id: dict[str, ScoredChunk] = {}

        normalized_dense = self._normalize(dense_results)
        normalized_sparse = self._normalize(sparse_results)

        for item in normalized_dense:
            combined_scores[item.chunk.chunk_id] += item.score * self.dense_weight
            chunk_by_id[item.chunk.chunk_id] = item

        for item in normalized_sparse:
            combined_scores[item.chunk.chunk_id] += item.score * self.sparse_weight
            chunk_by_id[item.chunk.chunk_id] = item

        merged = [ScoredChunk(chunk=chunk_by_id[chunk_id].chunk, score=score, source="hybrid") for chunk_id, score in combined_scores.items()]
        merged.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return merged

    def _normalize(self, results: list[ScoredChunk]) -> list[ScoredChunk]:
        if not results:
            return []
        scores = [result.score for result in results]
        minimum = min(scores)
        maximum = max(scores)
        if maximum == minimum:
            return [ScoredChunk(chunk=result.chunk, score=1.0, source=result.source) for result in results]

        normalized = []
        for result in results:
            score = (result.score - minimum) / (maximum - minimum)
            normalized.append(ScoredChunk(chunk=result.chunk, score=score, source=result.source))
        return normalized

    def _diversify_by_company(self, results: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        grouped: dict[str, list[ScoredChunk]] = defaultdict(list)
        for result in results:
            company = str(result.chunk.metadata.get("company", "unknown")).lower()
            grouped[company].append(result)

        for items in grouped.values():
            items.sort(key=lambda item: (-item.score, item.chunk.chunk_id))

        ordered_companies = sorted(grouped.keys())
        diversified: list[ScoredChunk] = []
        index = 0
        while len(diversified) < top_k and any(index < len(grouped[company]) for company in ordered_companies):
            for company in ordered_companies:
                if index < len(grouped[company]):
                    candidate = grouped[company][index]
                    if candidate.chunk.chunk_id not in {item.chunk.chunk_id for item in diversified}:
                        diversified.append(candidate)
                        if len(diversified) >= top_k:
                            break
            index += 1

        diversified.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return diversified[:top_k]