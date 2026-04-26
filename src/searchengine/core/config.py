from functools import lru_cache
from pydantic import BaseModel, Field


class Settings(BaseModel):
    app_name: str = Field(default="AI Search Engine")
    version: str = Field(default="0.1.0")
    chroma_path: str = Field(default="./data/chroma")
    chroma_collection: str = Field(default="search_chunks")
    embedding_model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    max_chunk_size: int = Field(default=900)
    chunk_overlap: int = Field(default=160)
    top_k_retrieval: int = Field(default=12)
    top_k_rerank: int = Field(default=5)
    llm_base_url: str | None = Field(default=None)
    llm_api_key: str | None = Field(default=None)
    llm_model: str | None = Field(default=None)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
