"""Orchestrate phase 2 retrieval calls."""

from __future__ import annotations

from config import DATA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, TOP_K_RETRIEVAL
from ingestion.corpus_loader import CorpusLoader
from ingestion.document_splitter import DocumentSplitter
from utils.logger import TranscriptLogger

from .hybrid_retriever import HybridRetriever
from .models import RetrievalResult


class RetrievalPipeline:
    def __init__(self, data_dir: str = DATA_DIR, logger: TranscriptLogger | None = None) -> None:
        self.data_dir = data_dir
        self.logger = logger or TranscriptLogger()
        self.retriever = HybridRetriever()
        self._ensure_index()

    def query(self, ticket_text: str, company: str | None) -> RetrievalResult:
        chunks = self.retriever.retrieve(ticket_text, company, top_k=TOP_K_RETRIEVAL)
        companies = sorted({str(chunk.chunk.metadata.get("company", "unknown")).lower() for chunk in chunks})
        self.logger.log("TOOL", f"retrieval company={company or 'any'} chunks={len(chunks)}")
        return RetrievalResult(chunks=chunks, retrieved_companies=companies)

    def _ensure_index(self) -> None:
        documents = CorpusLoader().load_all(self.data_dir)
        chunks = DocumentSplitter().split(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        self.retriever.vector_store.index_chunks(chunks)
        self.retriever.bm25_index.index(chunks)