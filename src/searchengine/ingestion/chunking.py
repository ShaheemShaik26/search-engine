from __future__ import annotations

from dataclasses import dataclass

from searchengine.utils.text import split_sentences, tokenize


@dataclass(slots=True)
class ChunkingPolicy:
    max_tokens: int = 220
    overlap_tokens: int = 40


class IntelligentChunker:
    def __init__(self, policy: ChunkingPolicy) -> None:
        self.policy = policy

    def chunk(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        if not sentences:
            return []

        chunks: list[str] = []
        current: list[str] = []
        current_token_count = 0

        for sentence in sentences:
            sentence_tokens = tokenize(sentence)
            sentence_len = len(sentence_tokens)
            if current and current_token_count + sentence_len > self.policy.max_tokens:
                chunks.append(" ".join(current).strip())
                current = self._build_overlap(current)
                current_token_count = sum(len(tokenize(part)) for part in current)
            current.append(sentence)
            current_token_count += sentence_len

        if current:
            chunks.append(" ".join(current).strip())
        return [chunk for chunk in chunks if chunk]

    def _build_overlap(self, sentences: list[str]) -> list[str]:
        if self.policy.overlap_tokens <= 0 or not sentences:
            return []
        overlap: list[str] = []
        token_count = 0
        for sentence in reversed(sentences):
            overlap.insert(0, sentence)
            token_count += len(tokenize(sentence))
            if token_count >= self.policy.overlap_tokens:
                break
        return overlap
