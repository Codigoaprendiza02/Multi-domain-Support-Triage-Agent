from dataclasses import dataclass
from typing import List, Optional


ESCALATION_KEYWORDS = {
    "fraud": [
        "fraud",
        "unauthorized charge",
        "stolen card",
        "identity theft",
        "chargeback",
        "didn't authorize",
        "did not authorize",
        "stole my",
        "stolen",
        "unauthorized transaction",
        "lost card",
        "card was used without my permission",
        "someone used my card",
        "card details leaked",
    ],
    "legal": [
        "lawsuit", "attorney", "court order", "subpoena", "gdpr", "legal", "regulatory", "compliance", "regulation", "law", "policy violation"
    ],
    "security": [
        "data breach", "hacked", "compromised account", "security incident", "security issue", "phishing", "malware", "account compromised"
    ],
    "billing_dispute": [
        "dispute charge", "wrong amount charged", "refund", "billing dispute", "overcharged", "double charged", "incorrect billing", "charge dispute"
    ],
    "account_locked": [
        "can't log in", "account suspended", "locked out", "cannot log in", "account locked", "account disabled", "unable to access account", "account blocked"
    ],
}


@dataclass
class EscalationResult:
    should_escalate: bool
    reason: Optional[str]
    matched_rules: List[str]


class EscalationClassifier:
    def classify(self, ticket_text: str, pii_matches: List[object]) -> EscalationResult:
        text = ticket_text.lower()
        # PII present -> escalate
        if pii_matches:
            return EscalationResult(should_escalate=True, reason="pii_detected", matched_rules=["pii"])

        matched = []
        for rule, kw_list in ESCALATION_KEYWORDS.items():
            for kw in kw_list:
                if kw in text:
                    matched.append(rule)
                    break

        if matched:
            # choose first as top-level reason
            return EscalationResult(should_escalate=True, reason=matched[0], matched_rules=matched)

        return EscalationResult(should_escalate=False, reason=None, matched_rules=[])
