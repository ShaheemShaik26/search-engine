import re
from datetime import datetime, timezone
from urllib.parse import urlparse


_whitespace_re = re.compile(r"\s+")
_sentence_re = re.compile(r"(?<=[.!?])\s+")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    parts = _sentence_re.split(cleaned)
    return [part.strip() for part in parts if part.strip()]


def compact_whitespace(text: str) -> str:
    return _whitespace_re.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9']+", text)]


def detect_source_name(url_or_path: str | None, fallback: str = "document") -> str:
    if not url_or_path:
        return fallback
    parsed = urlparse(url_or_path)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return parsed.netloc
    return url_or_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] or fallback


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
