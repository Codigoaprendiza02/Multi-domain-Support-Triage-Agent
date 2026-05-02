"""Token-window chunking for corpus documents."""

from __future__ import annotations

from .models import Chunk, Document


class DocumentSplitter:
    def split(self, documents: list[Document], chunk_size: int = 512, chunk_overlap: int = 64) -> list[Chunk]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be between 0 and chunk_size - 1")

        chunks: list[Chunk] = []
        step = chunk_size - chunk_overlap

        for document_index, document in enumerate(documents):
            tokens = self._tokenize(document.content)
            if not tokens:
                continue

            if len(tokens) <= chunk_size:
                chunks.append(
                    Chunk(
                        text=self._detokenize(tokens),
                        metadata=dict(document.metadata),
                        chunk_id=f"doc{document_index:04d}_chunk0000",
                    )
                )
                continue

            chunk_index = 0
            start = 0
            while start < len(tokens):
                window = tokens[start : start + chunk_size]
                if not window:
                    break
                chunks.append(
                    Chunk(
                        text=self._detokenize(window),
                        metadata=dict(document.metadata),
                        chunk_id=f"doc{document_index:04d}_chunk{chunk_index:04d}",
                    )
                )
                chunk_index += 1
                if start + chunk_size >= len(tokens):
                    break
                start += step

        return chunks

    def _tokenize(self, text: str) -> list[str]:
        return text.split()

    def _detokenize(self, tokens: list[str]) -> str:
        return " ".join(tokens).strip()