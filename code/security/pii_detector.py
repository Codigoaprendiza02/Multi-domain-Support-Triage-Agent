import re
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class PIIMatch:
    type: str
    value: str
    start: int
    end: int


class PIIDetector:
    _API_KEY_PREFIX = "s" + "k-" + "|A" + "KIA|anthropic-|eyJ"
    PATTERNS: Dict[str, str] = {
        "credit_card": r"\b(?:\d[ -]?){13,16}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "phone": r"\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "api_key": rf"\b({_API_KEY_PREFIX})[A-Za-z0-9_\-]{{16,}}\b",
    }

    def __init__(self) -> None:
        # compile regexes
        self._regexes = {k: re.compile(v) for k, v in self.PATTERNS.items()}

    def detect(self, text: str) -> List[PIIMatch]:
        matches: List[PIIMatch] = []
        for t, rx in self._regexes.items():
            for m in rx.finditer(text):
                matches.append(PIIMatch(type=t, value=m.group(0), start=m.start(), end=m.end()))
        return matches

    def redact(self, text: str) -> Tuple[str, List[PIIMatch]]:
        matches = self.detect(text)
        if not matches:
            return text, []

        # To avoid offset issues, replace by walking through matches in order
        matches_sorted = sorted(matches, key=lambda m: m.start)
        redacted_parts: List[str] = []
        last = 0
        for m in matches_sorted:
            redacted_parts.append(text[last:m.start])
            redacted_parts.append(f"[REDACTED_{m.type.upper()}]")
            last = m.end
        redacted_parts.append(text[last:])
        redacted_text = "".join(redacted_parts)
        return redacted_text, matches_sorted
