"""Lightweight CLI runner for all-in-agents.

Usage:
    python -m all_in_agents "Summarize README.md"           # single-shot
    python -m all_in_agents --model gpt-4o-mini             # REPL mode
    python -m all_in_agents --adapter anthropic --unsafe     # permissive mode
"""
from __future__ import annotations

import argparse
import asyncio
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="all_in_agents",
        description="Lightweight CLI for the all-in-agents framework.",
    )
    p.add_argument("goal", nargs="?", default=None, help="Goal to accomplish (omit for REPL mode)")
    p.add_argument("--model", "-m", default=None, help="Model identifier (e.g. gpt-4o, claude-sonnet-4-6)")
    p.add_argument("--adapter", "-a", choices=["openai", "anthropic"], default="openai", help="LLM adapter")
    p.add_argument("--workspace", "-w", default=".", help="Workspace root directory")
    p.add_argument("--unsafe", action="store_true", help="Permissive mode: approve all tool calls")
    p.add_argument("--system", "-s", default="", help="System prompt")
    return p


def _default_model(adapter: str) -> str:
    if adapter == "anthropic":
        return "claude-sonnet-4-6"
    return "gpt-4o"


async def _run_once(agent, goal: str) -> None:
    result = await agent.run(goal)
    print(result.final_answer)


async def _repl(agent) -> None:
    print("all-in-agents REPL (type 'exit' or Ctrl+C to quit)")
    while True:
        try:
            goal = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        goal = goal.strip()
        if not goal or goal.lower() in ("exit", "quit"):
            break
        result = await agent.run(goal)
        print(result.final_answer)
        print()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    model = args.model or _default_model(args.adapter)

    from .agents.base import Agent
    agent = Agent.quick(
        model=model,
        adapter=args.adapter,
        workspace=args.workspace,
        unsafe=args.unsafe,
        system=args.system,
    )

    if args.goal:
        asyncio.run(_run_once(agent, args.goal))
    else:
        asyncio.run(_repl(agent))

    return 0


if __name__ == "__main__":
    sys.exit(main())
