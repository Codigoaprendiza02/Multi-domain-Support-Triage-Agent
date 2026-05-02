"""Central configuration for the support triage agent."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K_RETRIEVAL = 12
RERANK_TOP_K = 6
MAX_TICKET_LENGTH = 2000
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "claude-sonnet-4-20250514"
# Preferred Gemini model; LLMClient auto-falls-back through flash / 2.5-flash on quota errors.
GEMINI_MODEL = "gemini-2.0-flash-lite"
RANDOM_SEED = 42
DATA_DIR = str(REPO_ROOT / "data")
OUTPUT_CSV = str(REPO_ROOT / "support_tickets" / "output.csv")
CHROMA_PERSIST_DIR = str(REPO_ROOT / ".chromadb")