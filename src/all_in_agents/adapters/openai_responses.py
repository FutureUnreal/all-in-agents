from __future__ import annotations

import json

from .base import GenerationOptions, LLMResponse, ToolCall
from .openai_utils import get_attr, parse_json_object, response_format_for_responses


def convert_responses_tools(tools: list[dict]) -> list[dict]:
    result = []
    for t in tools:
        result.append({
            "type": "function",
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("input_schema", t.get("parameters", {})),
        })
    return result


def convert_responses_input(messages: list[dict]) -> list[dict]:
    result = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")

        if isinstance(content, str):
            result.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            text_parts = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_result":
                    result.append({
                        "type": "function_call_output",
                        "call_id": block.get("tool_use_id", ""),
                        "output": block.get("content", ""),
                    })
                elif btype == "tool_use":
                    result.append({
                        "type": "function_call",
                        "call_id": block.get("id", ""),
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    })
            if text_parts:
                result.append({"role": role, "content": " ".join(text_parts)})
            continue

        result.append({"role": role, "content": content or ""})

    return result


def build_responses_kwargs(
    model_id: str,
    messages: list[dict],
    tools: list[dict],
    system: str,
    max_tokens: int,
    options: GenerationOptions,
) -> dict:
    kwargs: dict = dict(
        model=model_id,
        input=convert_responses_input(messages),
    )
    if system:
        kwargs["instructions"] = system
    if "max_output_tokens" not in options.extra:
        kwargs["max_output_tokens"] = max_tokens
    if tools:
        kwargs["tools"] = convert_responses_tools(tools)
    if options.temperature is not None:
        kwargs["temperature"] = options.temperature
    if options.top_p is not None:
        kwargs["top_p"] = options.top_p
    if options.response_format is not None:
        kwargs["text"] = {"format": response_format_for_responses(options.response_format)}
    if options.reasoning_effort is not None:
        kwargs["reasoning"] = {"effort": options.reasoning_effort}
    kwargs.update(options.extra)
    return kwargs


def parse_responses_response(resp) -> LLMResponse:
    content_text = getattr(resp, "output_text", None)
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for item in getattr(resp, "output", []) or []:
        item_type = get_attr(item, "type")
        if item_type == "function_call":
            args = parse_json_object(get_attr(item, "arguments") or "{}")
            tool_calls.append(ToolCall(
                id=get_attr(item, "call_id") or get_attr(item, "id") or "",
                name=get_attr(item, "name") or "",
                args=args,
            ))
            continue

        if item_type == "message":
            for block in get_attr(item, "content") or []:
                block_type = get_attr(block, "type")
                if block_type in ("output_text", "text"):
                    text = get_attr(block, "text")
                    if text:
                        text_parts.append(text)

    if content_text is None and text_parts:
        content_text = "\n".join(text_parts)

    usage = getattr(resp, "usage", None)
    return LLMResponse(
        content=content_text,
        tool_calls=tool_calls,
        input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
        output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
        stop_reason="tool_use" if tool_calls else "end_turn",
    )
