from dataclasses import dataclass
from typing import List


@dataclass
class InjectionResult:
    is_injection: bool
    matches: List[str]


class PromptInjectionDetector:
    COMMON_PATTERNS = [
        "ignore previous instructions",
        "you are now dan",
        "forget everything above",
        "system:",
        "override all rules",
        "do anything now",
    ]

    def detect(self, text: str) -> InjectionResult:
        t = text.lower()
        found: List[str] = []
        for p in self.COMMON_PATTERNS:
            if p in t:
                found.append(p)
        return InjectionResult(is_injection=bool(found), matches=found)
