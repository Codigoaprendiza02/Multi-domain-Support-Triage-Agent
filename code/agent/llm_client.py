from __future__ import annotations

import json
import importlib
import os
import random
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from dotenv import find_dotenv, load_dotenv

from config import LLM_MODEL, GEMINI_MODEL


@dataclass(slots=True)
class LLMResponse:
    text: str
    provider: str = "local"


class LLMClient:
    _queue_lock: threading.Lock = threading.Lock()
    _queue_condition: threading.Condition = threading.Condition(_queue_lock)
    _request_queue: deque[int] = deque()
    _next_request_id: int = 0
    _next_request_at: float = 0.0
    _model_cooldown_until: dict[str, float] = {}
    _response_cache: dict[str, str] = {}
    _request_count: int = 0
    _estimated_prompt_tokens_total: int = 0

    _MAX_OUTPUT_TOKENS: int = 160
    _MAX_RETRIES: int = 2
    _INTER_REQUEST_DELAY_SECONDS: float = 1.5

    def __init__(self, model: str = LLM_MODEL) -> None:
        self.model = model
        self.last_provider = "local"
        self.last_provider_error = ""
        load_dotenv(find_dotenv(usecwd=True), override=False)

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        company: str | None = None,
        issue: str = "",
        security_result: Any = None,
        retrieval_result: Any = None,
    ) -> str:
        provider_errors: list[str] = []
        prompt_key = self._cache_key(messages)

        cached = self._response_cache.get(prompt_key)
        if cached:
            self.last_provider = "cache"
            self.last_provider_error = ""
            print("[LLMClient] Cache hit for prompt; skipping live API call.")
            return cached

        # Pytest guard — never hit live API unless explicitly enabled
        if self._running_under_pytest() and os.environ.get("LLM_ENABLE_LIVE_API") != "1":
            self.last_provider = "local"
            self.last_provider_error = "pytest guard: using local provider"
            return self._generate_local(
                company=company,
                issue=issue,
                security_result=security_result,
                retrieval_result=retrieval_result,
            )

        # ── PRIMARY: Gemini (new google.genai SDK) ──────────────────────────────
        if os.environ.get("GEMINI_API_KEY"):
            try:
                text = self._generate_gemini_new(messages=messages)
                self.last_provider = "gemini"
                self.last_provider_error = ""
                self._response_cache[prompt_key] = text
                return text
            except Exception as exc:
                error_msg = f"gemini failed: {exc}"
                provider_errors.append(error_msg)
                print(f"[LLMClient] WARNING: {error_msg}")

        # ── FALLBACK: OpenAI ────────────────────────────────────────────────────
        if os.environ.get("OPENAI_API_KEY") and os.environ.get("LLM_ENABLE_CROSS_PROVIDER_FALLBACK") == "1":
            try:
                text = self._generate_openai(messages=messages)
                self.last_provider = "openai"
                self.last_provider_error = ""
                self._response_cache[prompt_key] = text
                return text
            except Exception as exc:
                error_msg = f"openai failed: {exc}"
                provider_errors.append(error_msg)
                print(f"[LLMClient] WARNING: {error_msg}")

        # ── LAST RESORT: local heuristic ────────────────────────────────────────
        self.last_provider = "local"
        self.last_provider_error = (
            " | ".join(provider_errors) if provider_errors else "no live provider configured"
        )
        print(f"[LLMClient] Falling back to local heuristic. Reason: {self.last_provider_error}")
        return self._generate_local(
            company=company,
            issue=issue,
            security_result=security_result,
            retrieval_result=retrieval_result,
        )

    # ── helpers ─────────────────────────────────────────────────────────────────

    def _running_under_pytest(self) -> bool:
        return bool(os.environ.get("PYTEST_CURRENT_TEST"))

    def _preferred_model_name(self) -> str:
        """Return the preferred Gemini model (env var overrides config constant)."""
        return os.environ.get("GEMINI_MODEL", GEMINI_MODEL)

    # ── new google.genai SDK (preferred) ────────────────────────────────────────

    def _generate_gemini_new(self, *, messages: list[dict[str, str]]) -> str:
        """Use the new google-genai SDK (google.genai) with retry on rate-limit errors."""
        try:
            genai = importlib.import_module("google.genai")
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is not installed. Run: pip install google-genai"
            ) from exc

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")

        client = genai.Client(api_key=api_key)
        prompt_variants = self._build_prompt_variants(messages)

        model_candidates = self._model_candidates()
        last_exc: Exception | None = None

        for model_name in model_candidates:
            cooldown_left = self._model_cooldown_left(model_name)
            if cooldown_left > 0:
                print(
                    f"[LLMClient] Skipping {model_name} for {cooldown_left:.1f}s "
                    "(provider cooldown)."
                )
                last_exc = RuntimeError(f"{model_name} in cooldown for {cooldown_left:.1f}s")
                continue
            print(f"[LLMClient] Trying Gemini model: {model_name}")
            text = self._try_model_with_retry(client, model_name, prompt_variants)
            if text is not None:
                print(f"[LLMClient] Success with model: {model_name}")
                return text
            # None means quota permanently exhausted for this model — try next
            print(f"[LLMClient] Quota exhausted for {model_name}, trying next model...")
            last_exc = RuntimeError(f"Quota exhausted for {model_name}")

        raise RuntimeError(f"All Gemini models quota-exhausted: {last_exc}") from last_exc

    def _try_model_with_retry(
        self,
        client: Any,
        model_name: str,
        prompt_variants: list[str],
        *,
        max_retries: int = _MAX_RETRIES,
        initial_wait: float = 2.0,
    ) -> str | None:
        """Try a single model with exponential backoff on transient rate-limit errors.

        Returns the response text on success, or None if quota is permanently exhausted.
        Raises RuntimeError for non-quota errors.
        """
        types = importlib.import_module("google.genai.types")
        prompt_count = max(1, len(prompt_variants))

        for attempt in range(max_retries):
            prompt_index = min(attempt, prompt_count - 1)
            prompt = prompt_variants[prompt_index]
            request_id = self._acquire_request_slot()
            try:
                prompt_tokens = self._estimate_tokens(prompt)
                type(self)._request_count += 1
                type(self)._estimated_prompt_tokens_total += prompt_tokens
                print(
                    f"[LLMClient][METRICS] requests={type(self)._request_count} "
                    f"prompt_tokens~={prompt_tokens} total_prompt_tokens~={type(self)._estimated_prompt_tokens_total}"
                )
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=self._MAX_OUTPUT_TOKENS,
                    ),
                )
                text = ""
                try:
                    text = response.text or ""
                except (AttributeError, IndexError, TypeError):
                    parts = getattr(response.candidates[0].content, "parts", []) if response.candidates else []
                    text = "".join(getattr(p, "text", "") for p in parts)

                if text.strip():
                    return text.strip()

                finish = ""
                try:
                    if response.candidates:
                        finish = str(response.candidates[0].finish_reason)
                except (AttributeError, IndexError, TypeError):
                    pass
                raise RuntimeError(f"Empty response (finish_reason={finish})")

            except Exception as exc:
                exc_str = str(exc)
                # Diagnostic: record the last provider error for debugging
                status_code = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
                resp_obj = getattr(exc, "response", None)
                retry_after = None
                try:
                    if resp_obj is not None:
                        # requests-like response
                        headers = getattr(resp_obj, "headers", {}) or {}
                        retry_after = headers.get("Retry-After") or headers.get("retry-after")
                except (AttributeError, TypeError):
                    retry_after = None

                retry_wait = self._extract_retry_delay_seconds(exc_str, retry_after)
                # Expose debug info on the client for diagnostics
                debug_info = (
                    f"{exc_str} | status={status_code} | retry_after={retry_after} "
                    f"| parsed_wait={retry_wait} | prompt_len={len(prompt)}"
                )
                self.last_provider_error = debug_info
                print(f"[LLMClient][DIAG] {debug_info}")
                is_quota = (
                    "429" in exc_str
                    or "RESOURCE_EXHAUSTED" in exc_str
                    or "quota" in exc_str.lower()
                )
                is_rate_limit = (
                    is_quota
                    or "rate" in exc_str.lower()
                    or "retry after" in exc_str.lower()
                    or "too many requests" in exc_str.lower()
                    or "503" in exc_str
                )
                is_not_found = "404" in exc_str or "NOT_FOUND" in exc_str

                if is_not_found:
                    # Model doesn't exist in this API version — skip permanently
                    return None

                if is_rate_limit:
                    hard_quota_exhausted = (
                        "limit: 0" in exc_str.lower() or "insufficient_quota" in exc_str.lower()
                    )
                    if hard_quota_exhausted:
                        self._set_model_cooldown(model_name, 60.0)
                        return None
                    if retry_wait is not None and retry_wait > 0:
                        self._set_model_cooldown(model_name, retry_wait)
                    if attempt < max_retries - 1:
                        # Prefer provider's wait value if present; otherwise use exponential backoff.
                        wait = retry_wait if retry_wait is not None else self._backoff_seconds(initial_wait, attempt)

                        # If provider gave a long Retry-After, respect it and set next request time
                        if retry_wait is not None:
                            type(self)._next_request_at = time.monotonic() + wait

                        print(
                            f"[LLMClient] Rate limit hit for {model_name}, waiting {wait:.1f}s "
                            f"(attempt {attempt + 1}/{max_retries})..."
                        )
                        time.sleep(wait)
                        continue
                    else:
                        # Exhausted retries — signal permanent quota for this model
                        return None

                # Non-quota error — propagate
                raise
            finally:
                self._release_request_slot(request_id)

        return None

    def _model_candidates(self) -> list[str]:
        """Return Gemini model names to try in preference order."""
        preferred = self._preferred_model_name()
        all_candidates = [
            preferred,
        ]
        if os.environ.get("LLM_ENABLE_MODEL_FALLBACKS") == "1":
            all_candidates.extend(["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-2.5-flash"])
        # Deduplicate while preserving order
        seen: set[str] = set()
        result: list[str] = []
        for m in all_candidates:
            if m not in seen:
                seen.add(m)
                result.append(m)
        return result

    def _backoff_seconds(self, initial_wait: float, attempt: int) -> float:
        base_wait = min(initial_wait * (2**attempt), 30.0)
        jitter = random.uniform(0.85, 1.15)
        return round(base_wait * jitter, 2)

    def _extract_retry_delay_seconds(self, exc_str: str, retry_after: str | None) -> float | None:
        if retry_after:
            try:
                return float(retry_after)
            except (TypeError, ValueError):
                pass

        patterns = [
            r"Please retry in\s+([0-9]+(?:\.[0-9]+)?)s",
            r"retryDelay['\"]?\s*:\s*['\"]([0-9]+(?:\.[0-9]+)?)s['\"]",
        ]
        for pattern in patterns:
            match = re.search(pattern, exc_str, flags=re.IGNORECASE)
            if not match:
                continue
            try:
                return float(match.group(1))
            except (TypeError, ValueError):
                continue

        return None

    def _set_model_cooldown(self, model_name: str, wait_seconds: float) -> None:
        until = time.monotonic() + max(0.0, wait_seconds)
        current = type(self)._model_cooldown_until.get(model_name, 0.0)
        if until > current:
            type(self)._model_cooldown_until[model_name] = until

    def _model_cooldown_left(self, model_name: str) -> float:
        until = type(self)._model_cooldown_until.get(model_name, 0.0)
        return max(0.0, until - time.monotonic())


    # ── OpenAI fallback ─────────────────────────────────────────────────────────

    def _generate_openai(self, *, messages: list[dict[str, str]]) -> str:
        try:
            openai_module = importlib.import_module("openai")
            OpenAI = getattr(openai_module, "OpenAI")
        except (ImportError, AttributeError) as exc:
            raise RuntimeError("openai is not installed") from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        client = OpenAI(api_key=api_key)
        model_name = self._openai_model_name()
        print(f"[LLMClient] Using OpenAI model: {model_name}")

        prompt = self._messages_to_prompt(messages)
        try:
            response = client.responses.create(
                model=model_name,
                input=prompt,
                temperature=0.0,
                max_output_tokens=self._MAX_OUTPUT_TOKENS,
            )
        except Exception as exc:
            raise RuntimeError(f"OpenAI API error for model '{model_name}': {exc}") from exc

        text = getattr(response, "output_text", "") or ""
        if not text.strip():
            raise RuntimeError("OpenAI returned an empty response")
        return text

    def _openai_model_name(self) -> str:
        configured = (self.model or "").strip()
        if configured.lower().startswith("gpt-"):
            return configured
        return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    # ── shared helpers ──────────────────────────────────────────────────────────

    def _messages_to_prompt(self, messages: list[dict[str, str]]) -> str:
        parts: list[str] = []
        for message in messages:
            role = message.get("role", "user").upper()
            content = self._squash_whitespace(message.get("content", ""))
            parts.append(f"[{role}]\n{content}")
        return "\n\n".join(parts)

    def _cache_key(self, messages: list[dict[str, str]]) -> str:
        return self._messages_to_prompt(messages)

    def _estimate_tokens(self, text: str) -> int:
        # Fast approximation for operational logging and quota troubleshooting.
        return max(1, len(text) // 4)

    def _build_prompt_variants(self, messages: list[dict[str, str]]) -> list[str]:
        base_prompt = self._messages_to_prompt(messages)
        variants: list[str] = [base_prompt]
        for budget in (12000, 8000, 5000):
            compact = self._compact_prompt(base_prompt, budget)
            if compact not in variants:
                variants.append(compact)
        return variants

    def _compact_prompt(self, prompt: str, max_chars: int) -> str:
        if len(prompt) <= max_chars:
            return prompt

        prompt = self._squash_whitespace(prompt)
        if len(prompt) <= max_chars:
            return prompt

        head = prompt[: max(0, max_chars - 24)].rstrip()
        return f"{head}\n[TRUNCATED FOR TOKEN BUDGET]"

    def _squash_whitespace(self, text: str) -> str:
        return " ".join(text.split())

    def _acquire_request_slot(self) -> int:
        with self._queue_condition:
            request_id = type(self)._next_request_id
            type(self)._next_request_id += 1
            type(self)._request_queue.append(request_id)

            while True:
                if type(self)._request_queue and type(self)._request_queue[0] == request_id:
                    wait_seconds = type(self)._next_request_at - time.monotonic()
                    if wait_seconds <= 0:
                        return request_id
                    self._queue_condition.wait(timeout=wait_seconds)
                    continue

                self._queue_condition.wait(timeout=0.05)

    def _release_request_slot(self, request_id: int) -> None:
        with self._queue_condition:
            if type(self)._request_queue and type(self)._request_queue[0] == request_id:
                type(self)._request_queue.popleft()
            else:
                try:
                    type(self)._request_queue.remove(request_id)
                except ValueError:
                    pass

            type(self)._next_request_at = time.monotonic() + self._INTER_REQUEST_DELAY_SECONDS
            self._queue_condition.notify_all()

    # ── local heuristic fallback ────────────────────────────────────────────────

    def _generate_local(
        self,
        *,
        company: str | None,
        issue: str,
        security_result: Any,
        retrieval_result: Any,
    ) -> str:
        issue_text = issue.lower()
        company_label = company if company else "general"

        should_escalate = bool(getattr(security_result, "should_escalate", False))
        if any(
            kw in issue_text
            for kw in ["fraud", "unauthorized", "breach", "lawsuit", "locked out", "cannot log in"]
        ):
            should_escalate = True

        if should_escalate:
            payload = {
                "status": "answered",  # will be overridden by triage_agent security check
                "product_area": self._product_area(company_label, issue_text),
                "response": "This case requires human review.",
                "justification": "The request matches escalation criteria or cannot be answered safely from corpus.",
                "reasoning_trace": "Security or policy trigger caused escalation.",
            }
            return json.dumps(payload)

        chunk_text = self._top_chunk_clean_text(retrieval_result)
        response = self._draft_response(company_label, chunk_text)
        payload = {
            "status": "answered",
            "product_area": self._product_area(company_label, issue_text),
            "response": response,
            "justification": "The answer was grounded in retrieved corpus content.",
            "reasoning_trace": "Grounded answer generated from retrieved chunks.",
        }
        return json.dumps(payload)

    def _product_area(self, company: str, issue_text: str) -> str:
        if "billing" in issue_text or "charge" in issue_text or "invoice" in issue_text:
            return "billing"
        if (
            "password" in issue_text
            or "login" in issue_text
            or "log in" in issue_text
            or "account" in issue_text
        ):
            return "account_access"
        if company.lower() == "visa":
            return "payments"
        if company.lower() == "claude":
            return "plans"
        return "assessments"

    def _top_chunk_clean_text(self, retrieval_result: Any) -> str:
        """Extract top chunk text, stripping any YAML frontmatter."""
        chunks = getattr(retrieval_result, "chunks", []) or []
        if not chunks:
            return ""
        top = chunks[0]
        chunk = getattr(top, "chunk", top)
        raw = str(getattr(chunk, "text", "")).strip()
        return self._strip_frontmatter(raw)

    def _strip_frontmatter(self, text: str) -> str:
        """Remove YAML frontmatter (--- ... ---) from text."""
        if not text.startswith("---"):
            return text
        lines = text.splitlines()
        end = None
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                end = i
                break
        if end is None:
            return text
        return "\n".join(lines[end + 1:]).lstrip("\n").strip()

    def _draft_response(self, company: str, chunk_text: str) -> str:
        if not chunk_text:
            return f"Please review the {company} support documentation for this request."
        paragraphs = [p.strip() for p in chunk_text.split("\n\n") if len(p.strip()) > 40]
        if paragraphs:
            return paragraphs[0][:400]
        first_sentence = chunk_text.split(".")[0].strip()
        return first_sentence if first_sentence else chunk_text[:200].strip()
