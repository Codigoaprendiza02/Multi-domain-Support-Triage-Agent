"""Environment secret validation."""

from __future__ import annotations

import os

from dotenv import load_dotenv, find_dotenv


PRIMARY_VARS = ["GEMINI_API_KEY"]
FALLBACK_VARS = ["OPENAI_API_KEY"]


def validate() -> None:
    load_dotenv(find_dotenv(usecwd=True), override=False)
    available = [name for name in PRIMARY_VARS + FALLBACK_VARS if os.environ.get(name)]
    if available:
        return

    joined = ", ".join(PRIMARY_VARS + FALLBACK_VARS)
    raise EnvironmentError(f"Missing required environment variable(s): {joined}")