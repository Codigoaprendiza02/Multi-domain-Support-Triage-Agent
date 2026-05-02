from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ParsedOutput:
    status: str
    product_area: str
    response: str
    justification: str
    resolution: str
    request_type: str
    reasoning_trace: str

    def to_csv_dict(self) -> dict[str, str]:
        return {
            "status": self.status,
            "product_area": self.product_area,
            "response": self.response,
            "resolution": self.resolution,
            "justification": self.justification,
            "request_type": self.request_type,
        }


class OutputParser:
    VALID_STATUSES = {"answered", "escalated"}
    VALID_REQUEST_TYPES = {"product_issue", "feature_request", "bug", "invalid"}

    def parse(self, raw_text: str) -> ParsedOutput:
        payload = self._load_payload(raw_text)
        if payload is None:
            return self._fallback_output("malformed_json")

        status = str(payload.get("status", "escalated")).strip().lower()
        if status not in self.VALID_STATUSES:
            status = "escalated"

        product_area = str(payload.get("product_area", "")).strip() or "general"
        response = str(payload.get("response", "")).strip() or "Unable to answer confidently from corpus."
        justification = str(payload.get("justification", "")).strip() or "The case was escalated for safety or grounding reasons."
        resolution = str(payload.get("resolution", "")).strip() or ""
        request_type = str(payload.get("request_type", "")).strip() or "product_issue"
        if request_type not in self.VALID_REQUEST_TYPES:
            request_type = "product_issue"
        reasoning_trace = str(payload.get("reasoning_trace", "")).strip() or "Decision trace unavailable."

        return ParsedOutput(
            status=status,
            product_area=product_area,
            response=response,
            justification=justification,
            resolution=resolution,
            request_type=request_type,
            reasoning_trace=reasoning_trace,
        )

    def _load_payload(self, raw_text: str) -> dict[str, Any] | None:
        text = raw_text.strip()
        if not text:
            return None

        # Strip markdown code fences that LLMs often add: ```json ... ``` or ``` ... ```
        if text.startswith("```"):
            lines = text.splitlines()
            # Remove opening fence line (```json or ```)
            lines = lines[1:]
            # Remove closing fence if present
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract the first {...} block
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return None
            return None

    def _fallback_output(self, reason: str) -> ParsedOutput:
        return ParsedOutput(
            status="escalated",
            product_area="general",
            response="Unable to process the request safely.",
            justification=f"Fallback triggered because {reason}.",
            resolution="",
            request_type="invalid",
            reasoning_trace=f"PARSER_FALLBACK: {reason}",
        )
