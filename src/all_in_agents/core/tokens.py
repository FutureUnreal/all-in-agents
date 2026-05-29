from __future__ import annotations

import json
from typing import Any


def estimate_text_tokens(value: Any) -> int:
    if value is None:
        return 0
    text = value if isinstance(value, str) else str(value)
    return max(1, len(text) // 4) if text else 0


def estimate_data_tokens(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return estimate_text_tokens(value)
    try:
        return estimate_text_tokens(json.dumps(value, ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError):
        return estimate_text_tokens(value)


def estimate_message_tokens(messages: list[dict]) -> int:
    total = 0
    for message in messages:
        total += estimate_text_tokens(message.get("role", ""))
        total += estimate_content_tokens(message.get("content", ""))
    return total


def estimate_content_tokens(content: Any) -> int:
    if isinstance(content, str):
        return estimate_text_tokens(content)
    if isinstance(content, list):
        return sum(estimate_data_tokens(block) for block in content)
    return estimate_data_tokens(content)
