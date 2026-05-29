from __future__ import annotations

import json

from .base import GenerationOptions, LLMResponse, LLMStreamEvent, ToolCall
from ..core.content import image_url_for_provider, is_file_block, is_image_block
from .openai_utils import get_attr, response_format_for_chat

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
                user_parts = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_result":
                        if user_parts:
                            result.append({"role": "user", "content": user_parts})
                            user_parts = []
                        result.append({
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": block.get("content", ""),
                        })
                    else:
                        part = _chat_user_content_part(block)
                        if part is not None:
                            user_parts.append(part)
                if user_parts:
                    result.append({"role": "user", "content": user_parts})
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


def _chat_user_content_part(block: dict) -> dict | None:
    btype = block.get("type")
    if btype in ("text", "input_text"):
        return {"type": "text", "text": str(block.get("text", ""))}
    if is_image_block(block):
        url = image_url_for_provider(block)
        if not url:
            return None
        image_url: dict = {"url": url}
        detail = block.get("detail")
        if detail:
            image_url["detail"] = detail
        return {"type": "image_url", "image_url": image_url}
    if is_file_block(block):
        return _chat_file_content_part(block)
    return None


def _chat_file_content_part(block: dict) -> dict:
    btype = block.get("type")
    if btype == "file_url":
        raise ValueError("OpenAI Chat Completions does not support file_url blocks; use file_base64/file_id or the Responses API")

    file_payload: dict = {}
    if btype == "file_base64":
        file_payload["file_data"] = block.get("data", "")
    elif btype == "file_id":
        file_payload["file_id"] = block.get("file_id", "")
    else:
        file_payload.update(block.get("file", {}) if isinstance(block.get("file"), dict) else {})

    filename = block.get("filename")
    if filename:
        file_payload["filename"] = filename
    return {"type": "file", "file": file_payload}


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


def _chunk_choices(chunk) -> list:
    choices = getattr(chunk, "choices", None)
    if choices is None and isinstance(chunk, dict):
        choices = chunk.get("choices")
    return list(choices or [])


async def stream_chat_response(stream):
    content_parts: list[str] = []
    tool_buffers: dict[int, dict] = {}
    finish_reason: str | None = None
    input_tokens = 0
    output_tokens = 0

    async for chunk in stream:
        usage = get_attr(chunk, "usage")
        if usage:
            input_tokens = get_attr(usage, "prompt_tokens") or input_tokens
            output_tokens = get_attr(usage, "completion_tokens") or output_tokens

        for choice in _chunk_choices(chunk):
            finish_reason = get_attr(choice, "finish_reason") or finish_reason
            delta = get_attr(choice, "delta")
            if delta is None:
                continue

            text_delta = get_attr(delta, "content")
            if text_delta:
                content_parts.append(text_delta)
                yield LLMStreamEvent(type="text_delta", delta=text_delta, raw=chunk)

            for tc in get_attr(delta, "tool_calls") or []:
                index = get_attr(tc, "index")
                if index is None:
                    index = len(tool_buffers)
                buf = tool_buffers.setdefault(index, {"id": "", "name": "", "arguments": ""})

                tc_id = get_attr(tc, "id")
                if tc_id:
                    buf["id"] = tc_id

                fn = get_attr(tc, "function")
                name_delta = get_attr(fn, "name") if fn is not None else None
                args_delta = get_attr(fn, "arguments") if fn is not None else None
                if name_delta:
                    buf["name"] += name_delta
                if args_delta:
                    buf["arguments"] += args_delta

                yield LLMStreamEvent(
                    type="tool_call_delta",
                    delta=args_delta or name_delta or "",
                    tool_call_delta={
                        "index": index,
                        "id": buf["id"],
                        "name": buf["name"],
                        "arguments_delta": args_delta or "",
                    },
                    raw=chunk,
                )

    tool_calls: list[ToolCall] = []
    for index in sorted(tool_buffers):
        buf = tool_buffers[index]
        try:
            args = json.loads(buf["arguments"] or "{}")
        except (json.JSONDecodeError, TypeError):
            args = {}
        tool_calls.append(ToolCall(id=buf["id"], name=buf["name"], args=args))

    response = LLMResponse(
        content="".join(content_parts) or None,
        tool_calls=tool_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        stop_reason="tool_use" if tool_calls else map_finish_reason(finish_reason),
    )
    yield LLMStreamEvent(type="message", response=response)
