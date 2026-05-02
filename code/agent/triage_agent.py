from __future__ import annotations

from dataclasses import dataclass

from config import MAX_TICKET_LENGTH
from retrieval.retrieval_pipeline import RetrievalPipeline
from security.output_sanitizer import OutputSanitizer
from security.security_pipeline import SecurityPipeline
from utils.logger import TranscriptLogger

from .context_builder import ContextBuilder
from .llm_client import LLMClient
from .output_parser import OutputParser, ParsedOutput
from .prompt_templates import build_system_prompt
from .reasoning_tracer import ReasoningTracer


@dataclass(slots=True)
class TriageResult:
    ticket_id: str
    status: str
    product_area: str
    response: str
    resolution: str
    justification: str
    request_type: str
    reasoning_trace: str

    def to_csv_dict(self) -> dict[str, str]:
        return {
            "ticket_id": self.ticket_id,
            "status": self.status,
            "product_area": self.product_area,
            "resolution": self.resolution,
            "response": self.response,
            "justification": self.justification,
            "request_type": self.request_type,
        }


class TriageAgent:
    def __init__(self, logger: TranscriptLogger | None = None) -> None:
        self.logger = logger or TranscriptLogger()
        self.security_pipeline = SecurityPipeline()
        self.retrieval_pipeline = RetrievalPipeline(logger=self.logger)
        self.context_builder = ContextBuilder()
        self.llm_client = LLMClient()
        self.output_parser = OutputParser()
        self.reasoning_tracer = ReasoningTracer()
        self.output_sanitizer = OutputSanitizer()

    def triage(self, ticket_id: str, company: str | None, issue: str) -> TriageResult:
        ticket_issue = issue[:MAX_TICKET_LENGTH]
        self.logger.log("TOOL", f"triage start ticket_id={ticket_id} company={company or 'None'}")

        security_result = self.security_pipeline.process(ticket_issue)
        retrieval_result = self.retrieval_pipeline.query(security_result.cleaned_text, company)

        system_prompt = build_system_prompt(company)
        context = self.context_builder.build(
            type("Ticket", (), {"ticket_id": ticket_id, "company": company, "issue": ticket_issue})(),
            retrieval_result.chunks,
            system_prompt=system_prompt,
        )

        raw_output = self.llm_client.generate(
            messages=context.messages,
            company=company,
            issue=ticket_issue,
            security_result=security_result,
            retrieval_result=retrieval_result,
        )
        provider_note = f"LLM provider used: {self.llm_client.last_provider}"
        if self.llm_client.last_provider_error:
            provider_note += f" | details: {self.llm_client.last_provider_error}"
        self.logger.log("TOOL", provider_note)
        if self.llm_client.last_provider == "local" and "429" in self.llm_client.last_provider_error:
            print(
                "WARNING: Live LLM provider returned HTTP 429 (quota/rate limit). "
                "Falling back to local response for this ticket."
            )
        parsed = self.output_parser.parse(raw_output)
        if parsed.status == "escalated" and parsed.justification.startswith("Fallback triggered"):
            # Log raw output for debugging when parsing fails
            self.logger.log("TOOL", f"parser_fallback ticket_id={ticket_id} raw_output_preview={raw_output[:200]!r}")

        if security_result.should_escalate:
            parsed = ParsedOutput(
                status="escalated",
                product_area=parsed.product_area,
                response="This case has been escalated for human review.",
                justification=security_result.reason or "Security checks required escalation.",
                resolution="escalated:security",
                request_type=self._infer_request_type(ticket_issue, security_result.reason or ""),
                reasoning_trace=parsed.reasoning_trace,
            )
        elif parsed.status == "answered":
            sanitizer_result = self.output_sanitizer.check(parsed.response, [item.chunk for item in retrieval_result.chunks])
            if not sanitizer_result.is_safe:
                parsed = ParsedOutput(
                    status="escalated",
                    product_area=parsed.product_area,
                    response="This case has been escalated for human review.",
                    justification="The drafted response was not safe or corpus-grounded.",
                    resolution="escalated:sanitizer",
                    request_type=self._infer_request_type(ticket_issue, "sanitizer"),
                    reasoning_trace=parsed.reasoning_trace,
                )

        # Ensure resolution is set for answered cases
        if parsed.status == "answered" and not getattr(parsed, "resolution", ""):
            parsed.resolution = "see response"

        reasoning_trace = self.reasoning_tracer.build(
            issue=ticket_issue,
            company=company,
            security_result=security_result,
            retrieval_result=retrieval_result,
            decision=parsed.status,
            response_source="retrieved chunks" if retrieval_result.chunks else "fallback",
        )

        self.logger.log("ASSISTANT", reasoning_trace)
        self.logger.log("TOOL", f"triage complete ticket_id={ticket_id} status={parsed.status}")

        return TriageResult(
            ticket_id=ticket_id,
            status=parsed.status,
            product_area=parsed.product_area,
            response=parsed.response,
            resolution=parsed.resolution,
            justification=parsed.justification,
            request_type=parsed.request_type,
            reasoning_trace=reasoning_trace,
        )

    def _infer_request_type(self, issue: str, hint: str) -> str:
        text = f"{issue} {hint}".lower()
        if not text.strip() or any(token in text for token in ["ignore previous instructions", "you are now dan", "system prompt"]):
            return "invalid"

        if any(token in text for token in ["feature", "request", "would like", "could you add", "please add", "enhancement"]):
            return "feature_request"

        if any(token in text for token in ["bug", "error", "broken", "fails", "not working", "cannot", "can't", "doesn't work"]):
            return "bug"

        return "product_issue"
