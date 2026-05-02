from __future__ import annotations

from config import LLM_MODEL


SYSTEM_PROMPT_TEMPLATE = """You are a support triage agent for {company}.
Your job: analyze the support ticket and produce a structured JSON response.


RULES:
1. Answer ONLY from the provided support corpus chunks below.
2. For the product_area field, always select the most specific and relevant label from the metadata of the retrieved corpus chunk(s) that supports your answer (e.g., 'screen', 'privacy', 'travel_support', etc.).
3. If information is insufficient or the case is sensitive, set status to "escalated".
4. NEVER fabricate URLs, phone numbers, or policies not in the corpus.
5. NEVER include PII from the ticket in your response.
6. Be concise and factual. No speculation.
7. Keep the final user-facing response under 100 tokens.

ESCALATION TRIGGERS (always escalate if any apply):
- Billing disputes, fraud, unauthorized transactions
- Account lockouts requiring identity verification
- Legal, regulatory, or compliance questions
- Information not found in corpus
- Security incidents




OUTPUT FORMAT (strict JSON, no markdown):
{{
    "status": "answered" | "escalated",
    "product_area": "<category>",
    "response": "<user-facing response>",
    "justification": "<1-3 sentence reasoning>",
    "request_type": "product_issue" | "feature_request" | "bug" | "invalid",
    "reasoning_trace": "<chain of thought — internal only>"
}}

Model: {model}
"""

USER_PROMPT_TEMPLATE = """TICKET:
{ticket_text}

CORPUS CHUNKS:
{context}
"""

FALLBACK_PROMPT_TEMPLATE = """If the retrieved corpus is insufficient or unsafe, return escalated output.
Use this fallback when the response cannot be grounded.
"""


def build_system_prompt(company: str | None) -> str:
    company_label = company if company else "all supported companies"
    return SYSTEM_PROMPT_TEMPLATE.format(company=company_label, model=LLM_MODEL)


def build_user_prompt(ticket_text: str, context: str) -> str:
    return USER_PROMPT_TEMPLATE.format(ticket_text=ticket_text, context=context)


def build_fallback_prompt() -> str:
    return FALLBACK_PROMPT_TEMPLATE
