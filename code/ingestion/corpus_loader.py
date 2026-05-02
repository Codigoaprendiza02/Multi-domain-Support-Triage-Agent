"""Load local corpus documents from the data directory."""

from __future__ import annotations

import logging
from pathlib import Path

from .metadata_extractor import MetadataExtractor
from .models import Document


logger = logging.getLogger(__name__)


class CorpusLoader:
    SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".html", ".htm", ".pdf"}

    def __init__(self, metadata_extractor: MetadataExtractor | None = None) -> None:
        self.metadata_extractor = metadata_extractor or MetadataExtractor()

    def load_all(self, data_dir: str) -> list[Document]:
        root = Path(data_dir)
        documents: list[Document] = []
        per_company: dict[str, int] = {}

        if not root.exists():
            logger.warning("Data directory does not exist: %s", root)
            return documents

        for file_path in sorted(path for path in root.rglob("*") if path.is_file()):
            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue

            try:
                content = self._read_file(file_path)
                if not content.strip():
                    continue
                metadata = self.metadata_extractor.extract(str(file_path), content, str(root))
                # strip frontmatter from stored document content so chunks don't include metadata headers
                try:
                    _, body = self.metadata_extractor._parse_frontmatter(content)
                except Exception:
                    body = content
                document_content = body if body and body.strip() else content
                company = str(metadata.get("company", "unknown"))
                documents.append(Document(content=document_content, metadata=metadata))
                per_company[company] = per_company.get(company, 0) + 1
            except Exception as exc:  # pragma: no cover - defensive guard for odd corpus files
                logger.warning("Skipping %s: %s", file_path, exc)

        for company, count in sorted(per_company.items()):
            logger.info("Loaded %s documents for company %s", count, company)

        return documents

    def _read_file(self, file_path: Path) -> str:
        extension = file_path.suffix.lower()
        if extension in {".md", ".markdown", ".txt"}:
            return file_path.read_text(encoding="utf-8")
        if extension in {".html", ".htm"}:
            return self._read_html(file_path)
        if extension == ".pdf":
            return self._read_pdf(file_path)
        return ""

    def _read_html(self, file_path: Path) -> str:
        html = file_path.read_text(encoding="utf-8", errors="ignore")
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            return soup.get_text("\n")
        except Exception:
            return html

    def _read_pdf(self, file_path: Path) -> str:
        try:
            from PyPDF2 import PdfReader
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("PyPDF2 is required to read PDF files") from exc

        reader = PdfReader(str(file_path))
        pages: list[str] = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)