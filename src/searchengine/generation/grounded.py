from __future__ import annotations

from searchengine.generation.llm import build_llm_client
from searchengine.schemas import SearchHit
from searchengine.utils.text import tokenize


class GroundedAnswerGenerator:
    def __init__(self) -> None:
        self.llm_client = build_llm_client()

    async def generate(self, query: str, hits: list[SearchHit]) -> tuple[str, bool]:
        context_blocks = self._build_context_blocks(hits)
        if not context_blocks:
            return "I could not find enough evidence in the indexed documents to answer that reliably.", False

        prompt = self._build_prompt(query, context_blocks)
        if self.llm_client:
            answer = await self.llm_client.generate(prompt)
        else:
            answer = self._extractive_answer(query, hits)

        grounded = self._is_grounded(answer, context_blocks)
        if not grounded:
            return "I could not verify that answer from the retrieved evidence.", False
        return answer, True

    def _build_context_blocks(self, hits: list[SearchHit]) -> list[str]:
        blocks = []
        for index, hit in enumerate(hits[:5], start=1):
            blocks.append(f"[{index}] {hit.source_name}: {hit.content}")
        return blocks

    def _build_prompt(self, query: str, context_blocks: list[str]) -> str:
        context = "\n\n".join(context_blocks)
        return (
            "Use only the context below to answer. Cite the supporting blocks explicitly.\n\n"
            f"Question: {query}\n\nContext:\n{context}\n\n"
            "If the evidence is insufficient, say you cannot determine the answer."
        )

    def _extractive_answer(self, query: str, hits: list[SearchHit]) -> str:
        query_tokens = set(tokenize(query))
        best_sentences: list[str] = []
        for hit in hits[:3]:
            sentences = [sentence.strip() for sentence in hit.content.split(".") if sentence.strip()]
            scored = []
            for sentence in sentences:
                sentence_tokens = set(tokenize(sentence))
                score = len(query_tokens & sentence_tokens)
                if score > 0:
                    scored.append((score, sentence))
            scored.sort(reverse=True)
            best_sentences.extend(sentence for _, sentence in scored[:2])
        if not best_sentences:
            return "I could not find a supported answer in the retrieved context."
        return ". ".join(best_sentences[:4]).strip() + "."

    def _is_grounded(self, answer: str, context_blocks: list[str]) -> bool:
        answer_tokens = set(tokenize(answer))
        context_tokens = set()
        for block in context_blocks:
            context_tokens.update(tokenize(block))
        return bool(answer_tokens & context_tokens)
