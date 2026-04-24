"""tiktoken-based exact token counting for all message types."""
from __future__ import annotations

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def count_message_tokens(message: dict) -> int:
    """Count tokens in a single chat message dict (role + content + overhead)."""
    role_tokens = count_tokens(message.get("role", ""))
    content = message.get("content", "")
    if isinstance(content, str):
        content_tokens = count_tokens(content)
    elif isinstance(content, list):
        content_tokens = sum(count_tokens(str(b)) for b in content)
    else:
        content_tokens = count_tokens(str(content))
    return role_tokens + content_tokens + 4  # per-message overhead
