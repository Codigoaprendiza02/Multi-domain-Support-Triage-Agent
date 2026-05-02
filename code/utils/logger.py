"""Transcript logger that appends to the shared home-directory log file."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


class TranscriptLogger:
    def __init__(self, log_path: str | None = None) -> None:
        self.log_path = Path(log_path) if log_path else Path.home() / "hackerrank_orchestrate" / "log.txt"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.touch(exist_ok=True)

    def log(self, role: str, content: str) -> None:
        timestamp = datetime.now().astimezone().isoformat()
        safe_role = role.encode("ascii", "backslashreplace").decode("ascii")
        safe_content = content.encode("ascii", "backslashreplace").decode("ascii")
        entry = f"[{timestamp}] [{safe_role}] {safe_content}\n"
        with self.log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(entry)