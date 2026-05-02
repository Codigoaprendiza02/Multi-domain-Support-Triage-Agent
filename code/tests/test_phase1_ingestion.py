from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.corpus_loader import CorpusLoader
from ingestion.document_splitter import DocumentSplitter
from ingestion.metadata_extractor import MetadataExtractor
from utils.logger import TranscriptLogger
from utils.secrets_validator import validate


CODE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = CODE_DIR.parent / "data"
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "sample_corpus"


def test_phase1_01_corpus_loader_loads_hackerrank_docs() -> None:
    docs = CorpusLoader().load_all(str(DATA_DIR / "hackerrank"))
    assert docs
    assert all(document.metadata["company"] == "hackerrank" for document in docs)


def test_phase1_02_corpus_loader_loads_claude_docs() -> None:
    docs = CorpusLoader().load_all(str(DATA_DIR / "claude"))
    assert docs
    assert all(document.metadata["company"] == "claude" for document in docs)


def test_phase1_03_corpus_loader_loads_visa_docs() -> None:
    docs = CorpusLoader().load_all(str(DATA_DIR / "visa"))
    assert docs
    assert all(document.metadata["company"] == "visa" for document in docs)


def test_phase1_04_corpus_loader_returns_non_empty_content() -> None:
    docs = CorpusLoader().load_all(str(DATA_DIR / "claude"))
    assert all(document.content.strip() for document in docs)


def test_phase1_05_corpus_loader_attaches_correct_company_metadata() -> None:
    docs = CorpusLoader().load_all(str(DATA_DIR))
    companies = {document.metadata["company"] for document in docs}
    assert {"hackerrank", "claude", "visa"}.issubset(companies)


def test_phase1_06_document_splitter_produces_chunks_within_limit() -> None:
    docs = CorpusLoader().load_all(str(DATA_DIR))
    chunks = DocumentSplitter().split(docs)
    assert chunks
    assert all(len(chunk.text.split()) <= 512 for chunk in chunks)


def test_phase1_07_document_splitter_preserves_source_metadata() -> None:
    docs = CorpusLoader().load_all(str(DATA_DIR))
    chunks = DocumentSplitter().split(docs)
    assert all("source_file" in chunk.metadata and "company" in chunk.metadata for chunk in chunks)


def test_phase1_08_document_splitter_chunk_overlap_is_correct() -> None:
    docs = CorpusLoader().load_all(str(FIXTURE_DIR / "claude"))
    chunks = DocumentSplitter().split(docs, chunk_size=12, chunk_overlap=4)
    assert len(chunks) > 1
    previous_tokens = chunks[0].text.split()
    current_tokens = chunks[1].text.split()
    assert previous_tokens[-4:] == current_tokens[:4]


def test_phase1_09_secrets_validator_raises_when_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # Mock load_dotenv to prevent loading from .env
    monkeypatch.setattr("utils.secrets_validator.load_dotenv", lambda *args, **kwargs: None)
    with pytest.raises(EnvironmentError):
        validate()


def test_phase1_10_secrets_validator_passes_when_api_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    validate()


def test_phase1_11_transcript_logger_creates_log_file(tmp_path: Path) -> None:
    log_path = tmp_path / "hackerrank_orchestrate" / "log.txt"
    logger = TranscriptLogger(str(log_path))
    logger.log("SYSTEM", "created")
    assert log_path.exists()
    assert log_path.read_text(encoding="utf-8").strip()


def test_phase1_12_transcript_logger_appends_on_second_call(tmp_path: Path) -> None:
    log_path = tmp_path / "hackerrank_orchestrate" / "log.txt"
    logger = TranscriptLogger(str(log_path))
    logger.log("SYSTEM", "first")
    first_size = log_path.stat().st_size
    logger.log("ASSISTANT", "second")
    second_size = log_path.stat().st_size
    assert second_size > first_size
    contents = log_path.read_text(encoding="utf-8")
    assert "first" in contents
    assert "second" in contents


def test_phase1_13_total_chunk_count_is_above_hundred() -> None:
    docs = CorpusLoader().load_all(str(DATA_DIR))
    chunks = DocumentSplitter().split(docs)
    assert len(chunks) > 100


def test_phase1_14_no_chunk_has_empty_text() -> None:
    docs = CorpusLoader().load_all(str(DATA_DIR))
    chunks = DocumentSplitter().split(docs)
    assert all(chunk.text.strip() for chunk in chunks)


def test_phase1_15_corpus_loader_skips_unsupported_file_types(tmp_path: Path) -> None:
    sample_dir = tmp_path / "corp"
    (sample_dir / "hackerrank").mkdir(parents=True)
    (sample_dir / "hackerrank" / "sample.md").write_text("---\ntitle: Sample\n---\nHello world", encoding="utf-8")
    (sample_dir / "hackerrank" / "image.png").write_bytes(b"not an image")
    docs = CorpusLoader().load_all(str(sample_dir))
    assert len(docs) == 1


def test_phase1_16_metadata_extractor_parses_frontmatter() -> None:
    extractor = MetadataExtractor()
    raw_text = "---\ntitle: Example\nsource_url: https://example.com/article\n---\nBody"
    metadata = extractor.extract(str(FIXTURE_DIR / "claude" / "sample.md"), raw_text, str(FIXTURE_DIR))
    assert metadata["company"] == "claude"
    assert metadata["title"] == "Example"
    assert metadata["url"] == "https://example.com/article"