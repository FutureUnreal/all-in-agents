from __future__ import annotations

import json

from .base import GenerationOptions, LLMResponse, ToolCall
from .openai_utils import response_format_for_chat

_STOP_REASON_MAP = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
}


def convert_chat_tools(tools: list[dict]) -> list[dict]:
    result = []
    for t in tools:
        result.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", t.get("parameters", {})),
            },
        })
    return result


def convert_chat_messages(messages: list[dict], system: str = "") -> list[dict]:
    result = []
    if system:
        result.append({"role": "system", "content": system})

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")

        if isinstance(content, str):
            result.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            if role == "user":
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        result.append({
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": block.get("content", ""),
                        })
                continue

            if role == "assistant":
                text_parts = []
                tool_calls = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "text":
                        text_parts.append(block.get("text", ""))
                    elif btype == "tool_use":
                        tool_calls.append({
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        })
                oai_msg: dict = {"role": "assistant"}
                oai_msg["content"] = " ".join(text_parts) if text_parts else None
                if tool_calls:
                    oai_msg["tool_calls"] = tool_calls
                result.append(oai_msg)
                continue

        result.append({"role": role, "content": content or ""})

    return result


def build_chat_kwargs(
    model_id: str,
    messages: list[dict],
    tools: list[dict],
    system: str,
    max_tokens: int,
    options: GenerationOptions,
) -> dict:
    kwargs: dict = dict(
        model=model_id,
        messages=convert_chat_messages(messages, system),
    )
    if "max_tokens" not in options.extra and "max_completion_tokens" not in options.extra:
        kwargs["max_completion_tokens"] = max_tokens
    if tools:
        kwargs["tools"] = convert_chat_tools(tools)
    if options.temperature is not None:
        kwargs["temperature"] = options.temperature
    if options.top_p is not None:
        kwargs["top_p"] = options.top_p
    if options.response_format is not None:
        kwargs["response_format"] = response_format_for_chat(options.response_format)
    if options.reasoning_effort is not None:
        kwargs["reasoning_effort"] = options.reasoning_effort
    kwargs.update(options.extra)
    return kwargs


def map_finish_reason(finish_reason: str | None) -> str:
    return _STOP_REASON_MAP.get(finish_reason or "", "end_turn")


def parse_chat_response(resp) -> LLMResponse:
    choice = resp.choices[0]
    message = choice.message
    content_text: str | None = message.content

    tool_calls: list[ToolCall] = []
    if message.tool_calls:
        for tc in message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, Exception):
                args = {}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, args=args))

    return LLMResponse(
        content=content_text,
        tool_calls=tool_calls,
        input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
        output_tokens=resp.usage.completion_tokens if resp.usage else 0,
        stop_reason=map_finish_reason(choice.finish_reason),
    )
