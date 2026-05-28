from __future__ import annotations

import json

from .base import GenerationOptions, LLMResponse, LLMStreamEvent, ToolCall
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


def _event_type(event) -> str:
    return get_attr(event, "type") or ""


def _event_key(event) -> str:
    return (
        get_attr(event, "item_id")
        or get_attr(event, "call_id")
        or str(get_attr(event, "output_index") or get_attr(event, "item_index") or len(str(event)))
    )


async def stream_responses_response(stream):
    content_parts: list[str] = []
    tool_buffers: dict[str, dict] = {}
    final_response = None

    async for event in stream:
        event_type = _event_type(event)

        if event_type in ("response.output_text.delta", "response.refusal.delta"):
            delta = get_attr(event, "delta") or ""
            if delta:
                content_parts.append(delta)
                yield LLMStreamEvent(type="text_delta", delta=delta, raw=event)
            continue

        if event_type == "response.function_call_arguments.delta":
            key = _event_key(event)
            buf = tool_buffers.setdefault(key, {"id": get_attr(event, "call_id") or "", "name": "", "arguments": ""})
            delta = get_attr(event, "delta") or ""
            buf["arguments"] += delta
            yield LLMStreamEvent(
                type="tool_call_delta",
                delta=delta,
                tool_call_delta={
                    "id": buf["id"],
                    "name": buf["name"],
                    "arguments_delta": delta,
                },
                raw=event,
            )
            continue

        if event_type in ("response.output_item.added", "response.output_item.done"):
            item = get_attr(event, "item")
            if get_attr(item, "type") == "function_call":
                key = get_attr(item, "id") or get_attr(item, "call_id") or _event_key(event)
                buf = tool_buffers.setdefault(key, {"id": "", "name": "", "arguments": ""})
                buf["id"] = get_attr(item, "call_id") or get_attr(item, "id") or buf["id"]
                buf["name"] = get_attr(item, "name") or buf["name"]
                arguments = get_attr(item, "arguments")
                if arguments:
                    buf["arguments"] = arguments
                yield LLMStreamEvent(
                    type="tool_call_delta",
                    tool_call_delta={
                        "id": buf["id"],
                        "name": buf["name"],
                        "arguments": buf["arguments"],
                    },
                    raw=event,
                )
            continue

        if event_type == "response.completed":
            response = get_attr(event, "response")
            if response is not None:
                final_response = parse_responses_response(response)

    if final_response is None:
        tool_calls = []
        for key in sorted(tool_buffers):
            buf = tool_buffers[key]
            tool_calls.append(ToolCall(
                id=buf["id"],
                name=buf["name"],
                args=parse_json_object(buf["arguments"] or "{}"),
            ))
        final_response = LLMResponse(
            content="".join(content_parts) or None,
            tool_calls=tool_calls,
            input_tokens=0,
            output_tokens=0,
            stop_reason="tool_use" if tool_calls else "end_turn",
        )

    yield LLMStreamEvent(type="message", response=final_response)
