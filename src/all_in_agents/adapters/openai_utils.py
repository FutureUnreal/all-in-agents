from __future__ import annotations

import json
from typing import Any

from .base import ConfigError


def normalize_api(api: str) -> str:
    normalized = api.replace("-", "_").lower()
    aliases = {
        "chat": "chat_completions",
        "chat_completion": "chat_completions",
        "chat_completions": "chat_completions",
        "responses": "responses",
        "response": "responses",
    }
    try:
        return aliases[normalized]
    except KeyError:
        raise ConfigError("OpenAIAdapter api must be 'chat_completions' or 'responses'")


def get_attr(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def parse_json_object(value: str) -> dict:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def response_format_for_chat(response_format: dict[str, Any]) -> dict[str, Any]:
    if response_format.get("type") != "json_schema":
        return response_format
    if "json_schema" in response_format:
        return response_format

    json_schema = {
        key: value
        for key, value in response_format.items()
        if key not in {"type"}
    }
    return {"type": "json_schema", "json_schema": json_schema}


def response_format_for_responses(response_format: dict[str, Any]) -> dict[str, Any]:
    if response_format.get("type") != "json_schema":
        return response_format
    if "json_schema" not in response_format:
        return response_format

    schema_value = response_format["json_schema"]
    if not isinstance(schema_value, dict):
        return response_format

    json_schema = dict(schema_value)
    json_schema.setdefault("type", "json_schema")
    return json_schema
