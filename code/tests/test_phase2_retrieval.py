from __future__ import annotations

from pathlib import Path

from ingestion.corpus_loader import CorpusLoader
from ingestion.document_splitter import DocumentSplitter
from ingestion.models import Chunk
from retrieval.bm25_index import BM25Index
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.models import ScoredChunk
from retrieval.reranker import Reranker
from retrieval.retrieval_pipeline import RetrievalPipeline
from retrieval.vector_store import VectorStore


CODE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = CODE_DIR.parent / "data"


def _build_chunks() -> list[Chunk]:
    documents = CorpusLoader().load_all(str(DATA_DIR))
    return DocumentSplitter().split(documents)


def test_phase2_01_vector_store_indexes_test_chunks_without_error(tmp_path: Path) -> None:
    chunks = _build_chunks()[:100]
    store = VectorStore(persist_dir=str(tmp_path / "chromadb"))
    store.index_chunks(chunks)
    assert store.is_indexed()


def test_phase2_02_vector_store_search_returns_no_more_than_top_k(tmp_path: Path) -> None:
    chunks = _build_chunks()[:50]
    store = VectorStore(persist_dir=str(tmp_path / "chromadb"))
    store.index_chunks(chunks)
    results = store.search("reset password", company=None, top_k=3)
    assert len(results) <= 3


def test_phase2_03_vector_store_search_returns_metadata_fields(tmp_path: Path) -> None:
    chunks = _build_chunks()[:50]
    store = VectorStore(persist_dir=str(tmp_path / "chromadb"))
    store.index_chunks(chunks)
    results = store.search("reset password", company=None, top_k=5)
    assert results
    assert all("company" in item.chunk.metadata and "source_file" in item.chunk.metadata for item in results)


def test_phase2_04_vector_store_company_filter_returns_only_requested_company(tmp_path: Path) -> None:
    chunks = _build_chunks()
    store = VectorStore(persist_dir=str(tmp_path / "chromadb"))
    store.index_chunks(chunks)
    results = store.search("password", company="hackerrank", top_k=10)
    assert results
    assert all(item.chunk.metadata["company"] == "hackerrank" for item in results)


def test_phase2_05_bm25_index_returns_keyword_relevant_results() -> None:
    chunks = _build_chunks()
    bm25 = BM25Index()
    bm25.index(chunks)
    results = bm25.search("unauthorized charge dispute refund", top_k=5)
    assert results
    joined = " ".join(item.chunk.text.lower() for item in results)
    assert any(keyword in joined for keyword in ["charge", "dispute", "refund"])


def test_phase2_06_hybrid_retriever_returns_combined_scores(tmp_path: Path) -> None:
    chunks = _build_chunks()
    store = VectorStore(persist_dir=str(tmp_path / "chromadb"))
    store.index_chunks(chunks)
    bm25 = BM25Index()
    bm25.index(chunks)
    retriever = HybridRetriever(vector_store=store, bm25_index=bm25, reranker=Reranker())
    results = retriever.retrieve("billing invoice payment failed", company=None, top_k=5)
    assert results
    assert all(item.score >= 0 for item in results)
    assert all(item.source in {"dense", "hybrid", "reranker"} for item in results)


def test_phase2_07_hybrid_retriever_deduplicates_overlap(tmp_path: Path) -> None:
    chunk = Chunk(text="Reset your password from account settings", metadata={"company": "claude", "source_file": "sample.md"}, chunk_id="dup-1")
    duplicate = Chunk(text="Reset your password from account settings", metadata={"company": "claude", "source_file": "sample.md"}, chunk_id="dup-1")
    store = VectorStore(persist_dir=str(tmp_path / "chromadb"))
    store.index_chunks([chunk, duplicate])
    bm25 = BM25Index()
    bm25.index([chunk, duplicate])
    retriever = HybridRetriever(vector_store=store, bm25_index=bm25, reranker=Reranker())
    results = retriever.retrieve("reset password", company="claude", top_k=10)
    ids = [item.chunk.chunk_id for item in results]
    assert len(ids) == len(set(ids))


def test_phase2_08_hybrid_retriever_with_none_company_returns_multiple_companies(tmp_path: Path) -> None:
    chunks = _build_chunks()
    store = VectorStore(persist_dir=str(tmp_path / "chromadb"))
    store.index_chunks(chunks)
    bm25 = BM25Index()
    bm25.index(chunks)
    retriever = HybridRetriever(vector_store=store, bm25_index=bm25, reranker=Reranker())
    results = retriever.retrieve("billing payment", company=None, top_k=8)
    companies = {item.chunk.metadata.get("company") for item in results}
    assert len(companies) >= 2


def test_phase2_09_reranker_reorders_top_chunks() -> None:
    chunks = [
        ScoredChunk(chunk=Chunk(text="alpha beta", metadata={"company": "hackerrank", "source_file": "a.md"}, chunk_id="a"), score=0.2, source="hybrid"),
        ScoredChunk(chunk=Chunk(text="alpha beta gamma", metadata={"company": "hackerrank", "source_file": "b.md"}, chunk_id="b"), score=0.21, source="hybrid"),
        ScoredChunk(chunk=Chunk(text="gamma delta", metadata={"company": "hackerrank", "source_file": "c.md"}, chunk_id="c"), score=0.19, source="hybrid"),
    ]
    reranked = Reranker().rerank("alpha gamma", chunks, top_k=3)
    assert [item.chunk.chunk_id for item in reranked] != [item.chunk.chunk_id for item in chunks]


def test_phase2_10_retrieval_pipeline_returns_non_empty_results_for_known_query() -> None:
    pipeline = RetrievalPipeline()
    result = pipeline.query("how do I cancel subscription", "Claude")
    assert result.chunks
    assert result.retrieved_companies == ["claude"]


def test_phase2_11_retrieval_is_deterministic() -> None:
    pipeline = RetrievalPipeline()
    first = pipeline.query("how to cancel subscription", "Claude")
    second = pipeline.query("how to cancel subscription", "Claude")
    assert [item.chunk.chunk_id for item in first.chunks] == [item.chunk.chunk_id for item in second.chunks]


def test_phase2_12_hybrid_retrieval_beats_dense_only_on_labeled_queries(tmp_path: Path) -> None:
    chunks = _build_chunks()
    store = VectorStore(persist_dir=str(tmp_path / "chromadb"))
    store.index_chunks(chunks)
    bm25 = BM25Index()
    bm25.index(chunks)
    hybrid = HybridRetriever(vector_store=store, bm25_index=bm25, reranker=Reranker())

    labeled_queries = [
        ("how do I reset my password", "claude"),
        ("unauthorized charge dispute refund", "visa"),
        ("how do I create a coding challenge", "hackerrank"),
        ("billing invoice payment failed", "claude"),
        ("account locked identity verification", "visa"),
    ]

    dense_score = 0.0
    hybrid_score = 0.0
    for query, expected_company in labeled_queries:
        dense_results = store.search(query, company=None, top_k=5)
        hybrid_results = hybrid.retrieve(query, company=None, top_k=5)
        dense_score += _mrr_at_5(dense_results, expected_company)
        hybrid_score += _mrr_at_5(hybrid_results, expected_company)

    assert hybrid_score >= dense_score


def test_phase2_13_chroma_persist_dir_created_on_first_index_run(tmp_path: Path) -> None:
    chunks = _build_chunks()[:20]
    persist_dir = tmp_path / "chromadb"
    store = VectorStore(persist_dir=str(persist_dir))
    store.index_chunks(chunks)
    assert persist_dir.exists()
    assert any(persist_dir.iterdir())


def test_phase2_14_reindexing_skipped_when_already_indexed(tmp_path: Path) -> None:
    chunks = _build_chunks()[:25]
    persist_dir = tmp_path / "chromadb"
    store = VectorStore(persist_dir=str(persist_dir))
    store.index_chunks(chunks)
    first_snapshot = (persist_dir / "vector_index.json").read_text(encoding="utf-8")
    store.index_chunks([])
    second_snapshot = (persist_dir / "vector_index.json").read_text(encoding="utf-8")
    assert first_snapshot == second_snapshot


def test_phase2_15_all_returned_chunks_have_non_empty_text(tmp_path: Path) -> None:
    chunks = _build_chunks()
    store = VectorStore(persist_dir=str(tmp_path / "chromadb"))
    store.index_chunks(chunks)
    bm25 = BM25Index()
    bm25.index(chunks)
    retriever = HybridRetriever(vector_store=store, bm25_index=bm25, reranker=Reranker())
    results = retriever.retrieve("billing invoice payment failed", company=None, top_k=8)
    assert results
    assert all(item.chunk.text.strip() for item in results)


def _mrr_at_5(results: list[ScoredChunk], expected_company: str) -> float:
    for index, item in enumerate(results[:5], start=1):
        if item.chunk.metadata.get("company") == expected_company:
            return 1 / index
    return 0.0