#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline Eval Harness — runs scripted test cases against the agent without external APIs."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from all_in_agents.adapters.base import LLMAdapter, LLMResponse, ToolCall
from all_in_agents.agents.base import Agent
from all_in_agents.tools.registry import Tool, ToolRegistry, ToolResponse, SideEffectLevel


# ---------------------------------------------------------------------------
# ScriptedAdapter
# ---------------------------------------------------------------------------

class ScriptedAdapter(LLMAdapter):
    model_id = "scripted"
    max_context_tokens = 32000

    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)

    async def generate(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
        max_tokens: int = 2048,
    ) -> LLMResponse:
        if not self._responses:
            return LLMResponse(
                content="[no more scripted responses]",
                tool_calls=[],
                input_tokens=0,
                output_tokens=0,
                stop_reason="end_turn",
            )

        raw = self._responses.pop(0)
        content: str | None = raw.get("content")
        stop_reason: str = raw.get("stop_reason", "end_turn")

        raw_tool_calls = raw.get("tool_calls") or []
        tool_calls = [
            ToolCall(
                id=tc.get("id", f"stub_{i}"),
                name=tc["name"],
                args=tc.get("args", tc.get("input", {})),
            )
            for i, tc in enumerate(raw_tool_calls)
        ]

        input_tokens = len(str(messages)) // 4
        output_tokens = len(content or "") // 4

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop_reason,
        )


# ---------------------------------------------------------------------------
# load_case
# ---------------------------------------------------------------------------

def load_case(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# build_registry
# ---------------------------------------------------------------------------

def build_registry(case: dict) -> ToolRegistry:
    registry = ToolRegistry()
    tool_stubs: dict[str, list[dict]] = case.get("tool_stubs", {})

    for tool_name, stub_responses in tool_stubs.items():
        responses = list(stub_responses)

        async def _execute(args: dict, run: Any, _responses=responses) -> ToolResponse:
            if not _responses:
                return ToolResponse("error", f"[stub exhausted for tool]")
            raw = _responses.pop(0)
            return ToolResponse(
                status=raw.get("status", "success"),
                content=raw.get("content", ""),
            )

        tool = Tool(
            name=tool_name,
            description=f"Stub tool: {tool_name}",
            input_schema={"type": "object", "properties": {}},
            side_effect_level=SideEffectLevel.READ_ONLY,
            execute=_execute,
        )
        registry.register(tool)

    return registry


# ---------------------------------------------------------------------------
# run_case
# ---------------------------------------------------------------------------

async def run_case(case: dict, work_dir: Path) -> tuple[dict, list[dict]]:
    llm = ScriptedAdapter(case.get("llm_responses", []))
    tools = build_registry(case)
    agent = Agent(llm, tools, run_dir=str(work_dir))

    shared = await agent.run(case["goal"])

    # Find the events.ndjson written by the agent
    events: list[dict] = []
    for ndjson_path in work_dir.rglob("events.ndjson"):
        with open(ndjson_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        break  # only first file

    return shared, events


# ---------------------------------------------------------------------------
# assert_case
# ---------------------------------------------------------------------------

def assert_case(case: dict, shared: dict, events: list[dict]) -> list[str]:
    assertions: dict = case.get("assertions", {})
    failures: list[str] = []

    # final_answer_contains
    if "final_answer_contains" in assertions:
        expected = assertions["final_answer_contains"]
        actual = shared.get("final_answer", "")
        if expected not in actual:
            failures.append(
                f"final_answer_contains: expected {expected!r} in {actual!r}"
            )

    # event_types
    if "event_types" in assertions:
        expected_types: list[str] = assertions["event_types"]
        actual_types = [ev.get("type") for ev in events]
        if actual_types != expected_types:
            failures.append(
                f"event_types mismatch:\n  expected: {expected_types}\n  actual:   {actual_types}"
            )

    # no_orphan_tool_use
    if assertions.get("no_orphan_tool_use"):
        tool_use_ids: set[str] = set()
        closed_ids: set[str] = set()
        for ev in events:
            ev_type = ev.get("type")
            payload = ev.get("payload", {})
            tid = payload.get("tool_use_id")
            if ev_type == "TOOL_USE" and tid:
                tool_use_ids.add(tid)
            elif ev_type in ("TOOL_RESULT", "TOOL_ABORTED") and tid:
                closed_ids.add(tid)
        orphans = tool_use_ids - closed_ids
        if orphans:
            failures.append(f"no_orphan_tool_use: orphaned tool_use_ids: {orphans}")

    # stop_reason_prefix
    if "stop_reason_prefix" in assertions:
        prefix = assertions["stop_reason_prefix"]
        terminal_event = None
        for ev in reversed(events):
            if ev.get("type") in ("RUN_STOPPED", "RUN_ABORTED"):
                terminal_event = ev
                break
        if terminal_event is None:
            failures.append(f"stop_reason_prefix: no RUN_STOPPED or RUN_ABORTED event found")
        else:
            reason = terminal_event.get("payload", {}).get("reason", "")
            if not reason.startswith(prefix):
                failures.append(
                    f"stop_reason_prefix: expected reason starting with {prefix!r}, got {reason!r}"
                )

    return failures


# ---------------------------------------------------------------------------
# main_async / main
# ---------------------------------------------------------------------------

async def main_async(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Offline eval harness for all-in-agent")
    parser.add_argument("cases", nargs="+", metavar="CASE_JSON", help="Path(s) to case JSON files")
    args = parser.parse_args(argv)

    total = 0
    passed = 0

    for case_path_str in args.cases:
        case_path = Path(case_path_str)
        total += 1
        case = load_case(case_path)
        name = case.get("name", case_path.stem)

        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            try:
                shared, events = await run_case(case, work_dir)
            except Exception as e:
                print(f"FAIL  {name}  [exception: {e}]")
                continue

        failures = assert_case(case, shared, events)
        if failures:
            print(f"FAIL  {name}")
            for f in failures:
                print(f"      - {f}")
        else:
            print(f"PASS  {name}")
            passed += 1

    print(f"\n{passed}/{total} passed")
    return 0 if passed == total else 1


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
