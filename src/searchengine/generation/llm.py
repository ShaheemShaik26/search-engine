from __future__ import annotations

from abc import ABC, abstractmethod

import httpx

from searchengine.core.config import settings


class LLMClient(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        raise NotImplementedError


class OpenAICompatibleLLMClient(LLMClient):
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    async def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Answer only from the provided context. If evidence is insufficient, say so."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        return data["choices"][0]["message"]["content"]


def build_llm_client() -> LLMClient | None:
    if settings.llm_base_url and settings.llm_api_key and settings.llm_model:
        return OpenAICompatibleLLMClient(settings.llm_base_url, settings.llm_api_key, settings.llm_model)
    return None
