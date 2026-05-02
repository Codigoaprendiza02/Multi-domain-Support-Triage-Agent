from __future__ import annotations

from dataclasses import dataclass
import sys
import types
from pathlib import Path

import tiktoken

from agent.context_builder import ContextBuilder
from agent.llm_client import LLMClient
from agent.output_parser import OutputParser
from agent.prompt_templates import build_system_prompt
from agent.triage_agent import TriageAgent
from retrieval.retrieval_pipeline import RetrievalPipeline


@dataclass
class Ticket:
    ticket_id: str
    company: str | None
    issue: str


def test_p4_01_context_builder_respects_token_budget():
    builder = ContextBuilder()
    pipeline = RetrievalPipeline()
    ticket = Ticket(ticket_id="T001", company="HackerRank", issue="I cannot submit my coding solution, it keeps saying time limit exceeded")
    chunks = pipeline.query(ticket.issue, ticket.company).chunks
    context = builder.build(ticket, chunks, system_prompt=build_system_prompt(ticket.company))
    enc = tiktoken.get_encoding("cl100k_base")
    total_tokens = sum(len(enc.encode(message["content"])) for message in context.messages)
    assert total_tokens <= 4096


def test_p4_02_context_builder_drops_lowest_scored_chunks():
    builder = ContextBuilder(total_budget=220)
    ticket = Ticket(ticket_id="T002", company="Claude", issue="How do I manage my plan billing and usage limits?")
    chunks = [
        type("Scored", (), {"chunk": type("Chunk", (), {"text": "short text one", "metadata": {"source_file": "a.md", "company": "claude"}, "chunk_id": "a"})(), "score": 0.1})(),
        type("Scored", (), {"chunk": type("Chunk", (), {"text": "short text two", "metadata": {"source_file": "b.md", "company": "claude"}, "chunk_id": "b"})(), "score": 0.9})(),
    ]
    context = builder.build(ticket, chunks, system_prompt=build_system_prompt(ticket.company))
    assert len(context.included_chunks) <= 2
    assert "b" in context.included_chunks


def test_p4_03_context_builder_includes_metadata():
    builder = ContextBuilder()
    ticket = Ticket(ticket_id="T003", company="Visa", issue="How can I dispute a transaction?")
    chunk = type("Scored", (), {"chunk": type("Chunk", (), {"text": "dispute a transaction at support", "metadata": {"source_file": "visa.md", "company": "visa"}, "chunk_id": "c"})(), "score": 1.0})()
    context = builder.build(ticket, [chunk], system_prompt=build_system_prompt(ticket.company))
    text = context.messages[1]["content"]
    assert "source: visa.md" in text
    assert "company: visa" in text


def test_p4_04_parser_parses_valid_json():
    parser = OutputParser()
    raw = '{"status":"answered","product_area":"billing","response":"Use the billing page.","justification":"Matched corpus.","reasoning_trace":"trace"}'
    parsed = parser.parse(raw)
    assert parsed.status == "answered"
    assert parsed.product_area == "billing"
    assert parsed.reasoning_trace == "trace"


def test_p4_05_parser_falls_back_on_malformed_json():
    parser = OutputParser()
    parsed = parser.parse("not json")
    assert parsed.status == "escalated"


def test_p4_06_parser_validates_status():
    parser = OutputParser()
    parsed = parser.parse('{"status":"maybe","product_area":"billing","response":"x","justification":"y","reasoning_trace":"z"}')
    assert parsed.status == "escalated"


def test_p4_07_reasoning_trace_present_in_parsed_output():
    parser = OutputParser()
    parsed = parser.parse('{"status":"answered","product_area":"billing","response":"x","justification":"y","reasoning_trace":"z"}')
    assert parsed.reasoning_trace == "z"


def test_p4_08_reasoning_trace_stripped_from_csv():
    parser = OutputParser()
    parsed = parser.parse('{"status":"answered","product_area":"billing","response":"x","justification":"y","reasoning_trace":"z"}')
    csv_dict = parsed.to_csv_dict()
    assert "reasoning_trace" not in csv_dict


def test_p4_09_llm_client_handles_empty_response_gracefully():
    client = LLMClient()
    result = client.generate(messages=[], company="HackerRank", issue="", security_result=type("S", (), {"should_escalate": False})(), retrieval_result=type("R", (), {"chunks": []})())
    assert result


def test_p4_09a_llm_client_uses_gemini_when_key_is_set(monkeypatch):
    """Verify LLMClient calls google.genai when GEMINI_API_KEY is set."""
    calls = {}

    expected_json = '{"status":"answered","product_area":"billing","response":"from gemini","justification":"ok","reasoning_trace":"trace"}'

    class FakeResponse:
        text = expected_json
        candidates = []

    class FakeModels:
        def generate_content(self, model, contents, config=None):
            calls["model"] = model
            calls["prompt"] = contents
            return FakeResponse()

    class FakeClient:
        models = FakeModels()

    def fake_client_constructor(api_key):
        calls["api_key"] = api_key
        return FakeClient()

    # Minimal types stub for GenerateContentConfig
    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            calls["config"] = kwargs

    fake_types = types.SimpleNamespace(GenerateContentConfig=FakeGenerateContentConfig)

    fake_genai = types.SimpleNamespace(
        Client=fake_client_constructor,
    )

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_ENABLE_LIVE_API", "1")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)
    # Also stub google.generativeai to avoid accidental import
    monkeypatch.setitem(sys.modules, "google.generativeai", types.SimpleNamespace())

    client = LLMClient()
    result = client.generate(
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "user"}],
        company="HackerRank",
        issue="x",
        security_result=type("S", (), {"should_escalate": False})(),
        retrieval_result=type("R", (), {"chunks": []})(),
    )

    assert calls.get("api_key") == "test-key"
    assert calls.get("model") == "gemini-test-model"
    assert "[SYSTEM]" in calls.get("prompt", "")
    assert "from gemini" in result


def test_p4_10_triage_agent_answers_faq():
    agent = TriageAgent()
    result = agent.triage("MANUAL_TEST_001", "HackerRank", "How do I download a certificate after completing a skill test?")
    assert result.status in {"answered", "escalated"}
    assert result.product_area
    assert result.justification


def test_p4_11_triage_agent_escalates_fraud():
    agent = TriageAgent()
    result = agent.triage("MANUAL_TEST_002", "Visa", "Someone stole my Visa card details and made purchases I didn't authorize.")
    assert result.status == "escalated"


def test_p4_12_triage_agent_output_is_grounded():
    agent = TriageAgent()
    result = agent.triage("MANUAL_TEST_003", "Claude", "What is included in Claude Pro plan?")
    assert result.response
    assert result.reasoning_trace


def test_p4_13_triage_agent_company_none_valid_output():
    agent = TriageAgent()
    result = agent.triage("MANUAL_TEST_004", None, "How do I reset my password?")
    assert result.status in {"answered", "escalated"}
    assert result.product_area


def test_p4_14_triage_agent_logs_every_call():
    agent = TriageAgent()
    result = agent.triage("MANUAL_TEST_005", "Claude", "How do I cancel my subscription?")
    assert result.ticket_id == "MANUAL_TEST_005"


def test_p4_14a_triage_agent_logs_provider_used():
    agent = TriageAgent()
    _result = agent.triage("MANUAL_TEST_005A", "Claude", "How do I cancel my subscription?")
    log_path = Path.home() / "hackerrank_orchestrate" / "log.txt"
    log_text = log_path.read_text(encoding="utf-8")
    assert "LLM provider used:" in log_text


def test_p4_15_system_prompt_injects_company_name():
    for company in ["HackerRank", "Claude", "Visa", None]:
        prompt = build_system_prompt(company)
        if company:
            assert company in prompt


def test_p4_16_product_area_non_empty():
    agent = TriageAgent()
    result = agent.triage("MANUAL_TEST_006", "HackerRank", "How do I submit a coding challenge?")
    assert result.product_area


def test_p4_17_justification_length():
    agent = TriageAgent()
    result = agent.triage("MANUAL_TEST_007", "Visa", "My card was charged twice")
    words = result.justification.split()
    # Justification should be non-empty and not excessively long (under 150 words)
    assert 1 <= len(words) <= 150
