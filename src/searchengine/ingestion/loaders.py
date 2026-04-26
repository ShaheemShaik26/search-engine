from __future__ import annotations

from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

from searchengine.utils.text import compact_whitespace, normalize_text


class DocumentLoader:
    def load_text(self, text: str) -> str:
        return normalize_text(text)

    def load_file(self, file_path: str) -> str:
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            reader = PdfReader(str(path))
            pages = [(page.extract_text() or "") for page in reader.pages]
            return normalize_text("\n".join(pages))
        return normalize_text(path.read_text(encoding="utf-8", errors="ignore"))

    async def load_url(self, url: str) -> str:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(["script", "style", "noscript"]):
            element.decompose()
        text = soup.get_text(separator=" ")
        return compact_whitespace(text)
