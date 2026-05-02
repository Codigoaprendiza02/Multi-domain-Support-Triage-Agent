from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ReasoningTracer:
    steps: list[str] = field(default_factory=list)

    def build(self, *, issue: str, company: str | None, security_result: Any, retrieval_result: Any, decision: str, response_source: str) -> str:
        lines = [
            "TICKET: [REDACTED]",
            f"COMPANY: {company or 'None'}",
            f"STEP 1 - Security Check: [{'PII found' if getattr(security_result, 'pii_matches', []) else 'PII not found'}] [{'Injection detected' if getattr(security_result, 'injection_detected', False) else 'Injection not detected'}]",
            f"STEP 2 - Escalation Rules: [{'Rules triggered' if getattr(security_result, 'should_escalate', False) else 'none'}]",
            f"STEP 3 - Retrieval: [{len(getattr(retrieval_result, 'chunks', []) or [])} chunks retrieved, top chunk: {self._top_source(retrieval_result)}]",
            f"STEP 4 - Corpus Coverage: [{self._coverage_label(retrieval_result)}]",
            f"STEP 5 - Decision: [{decision}] because [security and retrieval signals]",
            f"STEP 6 - Response Draft: [drafted from {response_source}]",
        ]
        trace = "\n".join(lines)
        self.steps = lines
        return trace

    def _top_source(self, retrieval_result: Any) -> str:
        chunks = getattr(retrieval_result, "chunks", []) or []
        if not chunks:
            return "none"
        top = chunks[0]
        chunk = getattr(top, "chunk", top)
        metadata = getattr(chunk, "metadata", {}) or {}
        return str(metadata.get("source_file", metadata.get("source", "unknown")))

    def _coverage_label(self, retrieval_result: Any) -> str:
        chunks = getattr(retrieval_result, "chunks", []) or []
        if not chunks:
            return "not covered"
        return "issue covered"
