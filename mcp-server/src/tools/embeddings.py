"""Embedding generation for hybrid RAG search.

Uses text-embedding-3-small (1536 dims, normalized) via the OpenAI API directly.
The embedding client is separate from the agent LLM client — embeddings always
go to api.openai.com, not to Anthropic's compatibility endpoint.

If EMBEDDING_API_KEY is not set, all embedding functions return None and the
search tools gracefully fall back to BM25-only (existing behaviour unchanged).
"""
from __future__ import annotations

import os
from typing import Optional

_client = None
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None
        try:
            from openai import OpenAI
            # Always use OpenAI directly for embeddings (not Anthropic compat endpoint)
            _client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")
        except ImportError:
            return None
    return _client


def embed_text(text: str) -> Optional[list[float]]:
    """Embed a single text string. Returns None if embedding unavailable."""
    client = _get_client()
    if client is None:
        return None
    try:
        response = client.embeddings.create(
            input=text[:8000],  # truncate to stay within token limit
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMS,
        )
        return response.data[0].embedding
    except Exception:
        return None


def embed_batch(texts: list[str]) -> list[Optional[list[float]]]:
    """Embed multiple texts in a single API call (up to 2048 per batch)."""
    client = _get_client()
    if client is None:
        return [None] * len(texts)
    try:
        truncated = [t[:8000] for t in texts]
        response = client.embeddings.create(
            input=truncated,
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMS,
        )
        results = [None] * len(texts)
        for item in response.data:
            results[item.index] = item.embedding
        return results
    except Exception:
        return [None] * len(texts)


def ticket_embed_text(title: str, description: str) -> Optional[list[float]]:
    return embed_text(f"{title} {description}")


def article_embed_text(title: str, content: str, tags: str) -> Optional[list[float]]:
    return embed_text(f"{title} {content} {tags}")


def embedding_to_pg(embedding: list[float]) -> str:
    """Convert embedding list to pgvector string format: '[0.1, 0.2, ...]'"""
    return "[" + ",".join(f"{v:.8f}" for v in embedding) + "]"
