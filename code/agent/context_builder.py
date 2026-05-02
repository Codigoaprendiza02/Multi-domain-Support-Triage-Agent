from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import MAX_TICKET_LENGTH

try:
    import tiktoken
except Exception:  # pragma: no cover - fallback when dependency is unavailable
    tiktoken = None


@dataclass(slots=True)
class ContextPackage:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_count: int = 0
    included_chunks: list[str] = field(default_factory=list)


class ContextBuilder:
    def __init__(self, total_budget: int = 4096, output_buffer: int = 400) -> None:
        self.total_budget = total_budget
        self.output_buffer = output_buffer
        self._encoding = None
        if tiktoken is not None:
            try:
                self._encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._encoding = None

    def build(self, ticket: Any, chunks: list[Any], system_prompt: str) -> ContextPackage:
        issue = self._truncate_text(str(getattr(ticket, "issue", "")))
        company = getattr(ticket, "company", None)
        ticket_id = getattr(ticket, "ticket_id", "unknown")

        sorted_chunks = sorted(chunks, key=lambda item: getattr(item, "score", 0.0), reverse=True)
        included_chunks: list[Any] = []
        context_lines: list[str] = []

        base_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"TICKET ID: {ticket_id}\nCOMPANY: {company or 'None'}\nISSUE: {issue}"},
        ]

        used_tokens = self._count_messages_tokens(base_messages)
        chunk_budget = max(0, self.total_budget - self.output_buffer - used_tokens)

        for index, scored_chunk in enumerate(sorted_chunks, start=1):
            formatted = self._format_chunk(index, scored_chunk)
            formatted_tokens = self._count_text_tokens(formatted)
            projected = self._count_text_tokens("\n\n".join(context_lines)) + formatted_tokens if context_lines else formatted_tokens
            if projected > chunk_budget:
                continue
            context_lines.append(formatted)
            included_chunks.append(scored_chunk)

        if not included_chunks and sorted_chunks:
            top_chunk = sorted_chunks[0]
            context_lines = [self._format_chunk(1, top_chunk)]
            included_chunks = [top_chunk]

        context_text = "\n\n".join(context_lines) if context_lines else ""
        if context_text:
            context_text += "\n\nchunks end here"
        else:
            context_text = "chunks end here"

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"TICKET ID: {ticket_id}\nCOMPANY: {company or 'None'}\nISSUE: {issue}\n\n{context_text}",
            },
        ]
        token_count = self._count_messages_tokens(messages)
        return ContextPackage(messages=messages, token_count=token_count, included_chunks=[getattr(chunk.chunk, "chunk_id", "") for chunk in included_chunks])

    def _format_chunk(self, index: int, scored_chunk: Any) -> str:
        chunk = getattr(scored_chunk, "chunk", scored_chunk)
        metadata = getattr(chunk, "metadata", {}) or {}
        source = metadata.get("source_file", metadata.get("source", "unknown"))
        company = metadata.get("company", "unknown")
        
        # Extract product_area from breadcrumbs (primary category)
        breadcrumbs = metadata.get("breadcrumbs", [])
        product_area = "unknown"
        if breadcrumbs:
            if isinstance(breadcrumbs, list) and len(breadcrumbs) > 0:
                product_area = breadcrumbs[0]
            elif isinstance(breadcrumbs, str):
                product_area = breadcrumbs.split("|")[0].strip()
        
        text = str(getattr(chunk, "text", "")).strip()
        return f"CORPUS EXCERPT [{index}] (source: {source}, company: {company}, product_area: {product_area}):\n{text}"

    def _truncate_text(self, text: str) -> str:
        if len(text) <= MAX_TICKET_LENGTH:
            return text
        return text[:MAX_TICKET_LENGTH]

    def _count_messages_tokens(self, messages: list[dict[str, str]]) -> int:
        return sum(self._count_text_tokens(message.get("content", "")) for message in messages)

    def _count_text_tokens(self, text: str) -> int:
        if self._encoding is not None:
            return len(self._encoding.encode(text))
        return max(1, len(text) // 4)
