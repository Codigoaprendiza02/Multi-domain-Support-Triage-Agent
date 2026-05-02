import re
from dataclasses import dataclass
from typing import List, Tuple
from .pii_detector import PIIDetector


URL_RE = re.compile(r"https?://[\w\-./?=#%&]+", flags=re.IGNORECASE)


@dataclass
class SanitizationResult:
    is_safe: bool
    issues: List[str]
    flagged_phrases: List[str]


class OutputSanitizer:
    def __init__(self) -> None:
        self._pii = PIIDetector()

    def check(self, response: str, retrieved_chunks: List[object]) -> SanitizationResult:
        issues: List[str] = []
        flagged: List[str] = []

        # 1) PII literal checks
        pii_matches = self._pii.detect(response)
        if pii_matches:
            issues.append("pii_in_response")
            flagged.extend([m.value for m in pii_matches])

        # 2) Hallucinated URLs: any URL not present in at least one chunk
        urls = URL_RE.findall(response)
        if urls:
            corpus_text = "\n".join(getattr(c, "text", "") for c in retrieved_chunks or [])
            for u in urls:
                if u not in corpus_text:
                    issues.append("hallucinated_url")
                    flagged.append(u)

        is_safe = len(issues) == 0
        return SanitizationResult(is_safe=is_safe, issues=issues, flagged_phrases=flagged)
