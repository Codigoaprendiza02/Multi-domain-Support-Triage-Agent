"""Metadata extraction for corpus documents."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


class MetadataExtractor:
    """Extracts metadata from markdown frontmatter and file paths."""

    def extract(self, file_path: str, raw_text: str, data_root: str) -> dict[str, Any]:
        path = Path(file_path)
        metadata, _ = self._parse_frontmatter(raw_text)
        company = self._infer_company(path, data_root)
        source_file = str(path.relative_to(Path(data_root))) if self._is_under(path, data_root) else path.name
        url = metadata.get("final_url") or metadata.get("source_url") or ""

        extracted = {
            "company": company,
            "source_file": source_file.replace("\\", "/"),
            "url": url,
        }
        extracted.update(metadata)
        return extracted

    def _parse_frontmatter(self, text: str) -> tuple[dict[str, Any], str]:
        if not text.startswith("---"):
            return {}, text

        lines = text.splitlines()
        if len(lines) < 3:
            return {}, text

        end_index = None
        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                end_index = index
                break

        if end_index is None:
            return {}, text

        metadata: dict[str, Any] = {}
        for line in lines[1:end_index]:
            match = re.match(r'^(?P<key>[A-Za-z0-9_\-]+):\s*(?P<value>.*)$', line)
            if not match:
                continue
            key = match.group("key").strip()
            value = match.group("value").strip().strip('"')
            metadata[key] = value

        body = "\n".join(lines[end_index + 1 :]).lstrip("\n")
        return metadata, body

    def _infer_company(self, path: Path, data_root: str) -> str:
        root = Path(data_root).resolve()
        try:
            path.resolve().relative_to(root)
        except ValueError:
            return path.parts[-2].lower() if len(path.parts) > 1 else "unknown"

        if root.name.lower() in {"hackerrank", "claude", "visa"}:
            return root.name.lower()

        relative = path.resolve().relative_to(root)
        return relative.parts[0].lower() if relative.parts else root.name.lower()

    def _is_under(self, path: Path, data_root: str) -> bool:
        try:
            path.resolve().relative_to(Path(data_root).resolve())
            return True
        except ValueError:
            return False