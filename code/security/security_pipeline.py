from dataclasses import dataclass
from typing import List, Optional
from .pii_detector import PIIDetector
from .prompt_injection import PromptInjectionDetector
from .escalation_rules import EscalationClassifier, EscalationResult


@dataclass
class SecurityPipelineResult:
    should_escalate: bool
    reason: Optional[str]
    cleaned_text: str
    pii_matches: List[object]
    injection_detected: bool
    matched_injection_patterns: List[str]


class SecurityPipeline:
    def __init__(self) -> None:
        self._pii = PIIDetector()
        self._inject = PromptInjectionDetector()
        self._escalate = EscalationClassifier()

    def process(self, text: str) -> SecurityPipelineResult:
        # 1) redact PII
        redacted_text, pii_matches = self._pii.redact(text)

        # 2) detect prompt injection
        inj = self._inject.detect(text)

        # 3) classify escalation
        esc: EscalationResult = self._escalate.classify(text, pii_matches)

        should_escalate = esc.should_escalate
        reason = esc.reason
        # If injection detected, escalate for safety
        if inj.is_injection:
            should_escalate = True
            reason = reason or "injection_detected"

        return SecurityPipelineResult(
            should_escalate=should_escalate,
            reason=reason,
            cleaned_text=redacted_text,
            pii_matches=pii_matches,
            injection_detected=inj.is_injection,
            matched_injection_patterns=inj.matches,
        )
