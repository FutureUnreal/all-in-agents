from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from typing import TYPE_CHECKING, Callable, Awaitable, Any, Iterable

from ..core.flow import Flow
from ..core.node import BaseNode
from ..core.run import BudgetExceededError, Budget, LoopDetectedError, Run, RunResult
from ..history.manager import HistoryManager
from ..history.store import FileEventStore
from ..tools.policy import ToolPolicy
from ..history.compactor import CompactionStrategy
from ..agents.harness import build_system_prompt
from ..utils import make_ulid as _make_ulid, iso_now as _iso_now

if TYPE_CHECKING:
    from ..adapters.base import LLMAdapter, LLMResponse
    from ..tools.registry import ToolRegistry


def _tool_sig(name: str, args: dict) -> str:
    payload = json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"{name}:{h}"


def _normalize_skill_selection(skills: Iterable[str] | str | None) -> tuple[bool, tuple[str, ...] | None]:
    if skills is None:
        return False, None
    if isinstance(skills, str):
        if skills == "all":
            return True, None
        return False, (skills,)
    return False, tuple(skills)


# Deprecated: use LLMCallNode + ToolDispatchNode
class ReActNode(BaseNode):
    async def exec(self, prep: dict) -> "LLMResponse":
        run: Run = prep["run"]
        run.check_budget("llm_call")
        return await prep["llm"].generate(
            messages=prep["messages"],
            tools=prep["tools"],
            system=prep["system"],
        )

    async def prep(self, shared: dict) -> dict:
        run: Run = shared["run"]
        return {
            "messages": shared["history"].get_messages(),
            "tools": shared["tools"].get_schemas(policy=run.tool_policy),
            "run": run,
            "llm": shared["llm"],
            "system": shared.get("system", ""),
        }

    async def post(self, shared: dict, resp: "LLMResponse") -> str:
        run: Run = shared["run"]
        store: FileEventStore = shared.get("store")
        history: HistoryManager = shared["history"]
        tools: "ToolRegistry" = shared["tools"]
        turn_id = _make_ulid()

        if store:
            await store.append(run.run_id, "ASSISTANT_MESSAGE", {
                "content": resp.content,
                "tool_calls": [{"id": tc.id, "name": tc.name, "args": tc.args} for tc in resp.tool_calls],
                "stop_reason": resp.stop_reason,
            })

        if resp.stop_reason == "end_turn" or not resp.tool_calls:
            shared["final_answer"] = resp.content or ""
            return "done"

        history.add("assistant", resp.content or "")

        for tc in resp.tool_calls:
            run.check_budget("tool_call", _tool_sig(tc.name, tc.args))
            if store:
                await store.append_tool_use(run.run_id, turn_id=turn_id, tool_use_id=tc.id, name=tc.name, args=tc.args)
            try:
                result = await tools.execute(tc.name, tc.args, run)
            except Exception as e:
                if store:
                    await store.append_tool_aborted(run.run_id, turn_id=turn_id, tool_use_id=tc.id, name=tc.name, reason=str(e), error_class=type(e).__name__)
                raise

            if store:
                await store.append_tool_result(run.run_id, turn_id=turn_id, tool_use_id=tc.id, name=tc.name, status=result.status, content=result.content)

            history.add_tool_result(tc.id, result)

        if history.needs_compression():
            await history.compress(shared["llm"])
            if store:
                await store.append(run.run_id, "MEMORY_UPDATED", {"summary": history._summary[:200]})

        return "continue"


class LLMCallNode(BaseNode):
    async def prep(self, shared: dict) -> dict:
        run: Run = shared["run"]
        history: HistoryManager = shared["history"]
        llm: "LLMAdapter" = shared["llm"]
        max_tokens = None
        if run.budget.max_input_tokens_per_call > 0:
            max_tokens = min(llm.max_context_tokens, run.budget.max_input_tokens_per_call)
        max_output_tokens = run.budget.max_output_tokens_per_call
        if max_output_tokens <= 0:
            max_output_tokens = 2048
        return {
            "messages": history.get_messages(max_tokens=max_tokens),
            "tools": shared["tools"].get_schemas(policy=run.tool_policy),
            "run": run,
            "llm": llm,
            "system": shared.get("system", ""),
            "max_output_tokens": max_output_tokens,
        }

    async def exec(self, prep: dict) -> "LLMResponse":
        run: Run = prep["run"]
        run.check_budget("llm_call")
        return await prep["llm"].generate(
            messages=prep["messages"],
            tools=prep["tools"],
            system=prep["system"],
            max_tokens=prep["max_output_tokens"],
        )

    async def post(self, shared: dict, resp: "LLMResponse") -> str:
        run: Run = shared["run"]
        store: FileEventStore = shared.get("store")
        history: HistoryManager = shared["history"]

        run.record_llm_usage(
            getattr(resp, "input_tokens", 0) or 0,
            getattr(resp, "output_tokens", 0) or 0,
        )

        if store:
            await store.append(run.run_id, "ASSISTANT_MESSAGE", {
                "content": resp.content,
                "tool_calls": [{"id": tc.id, "name": tc.name, "args": tc.args} for tc in resp.tool_calls],
                "stop_reason": resp.stop_reason,
            })

        if resp.stop_reason == "end_turn" or not resp.tool_calls:
            shared["final_answer"] = resp.content or ""
            return "done"

        turn_id = _make_ulid()
        shared["turn_id"] = turn_id
        # Write into shared (HC-5: Flow.copy causes node instance fields to not persist)
        shared["llm_response"] = resp
        history.add_assistant_tool_calls(resp.content or "" if resp.content else None, resp.tool_calls, turn_id=turn_id)
        return "dispatch_tools"


class ToolDispatchNode(BaseNode):
    def __init__(self, max_concurrency: int = 4, on_tool_result: Callable | None = None):
        super().__init__()
        self.max_concurrency = max_concurrency
        self.on_tool_result = on_tool_result

    async def prep(self, shared: dict) -> "LLMResponse | None":
        return shared.get("llm_response")

    async def exec(self, resp: "LLMResponse | None") -> "LLMResponse | None":
        # exec cannot access shared; tool execution must happen in post
        return resp

    async def post(self, shared: dict, resp: "LLMResponse | None") -> str:
        if resp is None:
            shared["final_answer"] = shared.get("final_answer", "")
            return "done"

        run: Run = shared["run"]
        store: FileEventStore = shared.get("store")
        history: HistoryManager = shared["history"]
        tools: "ToolRegistry" = shared["tools"]
        turn_id: str = shared.get("turn_id") or _make_ulid()

        from ..tools.policy import SideEffectLevel

        # Write TOOL_USE events for all tool calls first
        for tc in resp.tool_calls:
            run.check_budget("tool_call", _tool_sig(tc.name, tc.args))
            if store:
                await store.append_tool_use(
                    run.run_id, turn_id=turn_id,
                    tool_use_id=tc.id, name=tc.name, args=tc.args,
                )

        # Partition into concurrent-safe vs sequential
        concurrent_safe = []
        sequential = []
        for tc in resp.tool_calls:
            tool = tools.get_tool(tc.name)
            level = tool.side_effect_level if tool else SideEffectLevel.DANGEROUS
            if level in (SideEffectLevel.READ_ONLY, SideEffectLevel.NETWORK):
                concurrent_safe.append(tc)
            else:
                sequential.append(tc)

        # Results dict keyed by tool_use_id to preserve original order
        results: dict[str, Any] = {}

        # Execute concurrent-safe tools in parallel
        if concurrent_safe:
            sem = asyncio.Semaphore(self.max_concurrency)

            async def _run_one(tc):
                async with sem:
                    try:
                        return tc.id, await tools.execute(tc.name, tc.args, run)
                    except Exception as e:
                        return tc.id, e

            gathered = await asyncio.gather(*[_run_one(tc) for tc in concurrent_safe])
            for tc_id, result in gathered:
                results[tc_id] = result

        # Execute sequential tools one by one
        for tc in sequential:
            try:
                results[tc.id] = await tools.execute(tc.name, tc.args, run)
            except Exception as e:
                results[tc.id] = e

        # Write results in original tool_calls order
        for tc in resp.tool_calls:
            result = results.get(tc.id)
            if isinstance(result, Exception):
                if store:
                    await store.append_tool_aborted(
                        run.run_id, turn_id=turn_id, tool_use_id=tc.id,
                        name=tc.name, reason=str(result), error_class=type(result).__name__,
                    )
                # Synthesize an error ToolResponse so history stays consistent
                from ..tools.registry import ToolResponse
                result = ToolResponse("error", str(result), type(result).__name__)
            else:
                # Redact if configured
                if store:
                    result = store.redact_tool_response(tc.name, result)
                if store:
                    await store.append_tool_result(
                        run.run_id, turn_id=turn_id, tool_use_id=tc.id,
                        name=tc.name, status=result.status, content=result.content,
                    )
                run.record_tool_result(tc.name, result.status)

            history.add_tool_result(tc.id, result, turn_id=turn_id)

            if self.on_tool_result is not None:
                try:
                    cb_result = self.on_tool_result({"tool_use_id": tc.id, "name": tc.name, "result": result})
                    if asyncio.iscoroutine(cb_result):
                        await cb_result
                except Exception:
                    pass  # callback errors must not interrupt main flow

        if history.needs_compression():
            await history.compress(shared["llm"])
            run.record_compression()
            if store:
                await store.append(run.run_id, "MEMORY_UPDATED", {"summary": history._summary[:200]})

        return "continue"


class Agent:
    def __init__(
        self,
        llm: "LLMAdapter",
        tools: "ToolRegistry",
        budget: Budget | None = None,
        run_dir: str = "./runs",
        system: str = "",
        *,
        tool_policy: ToolPolicy | None = None,
        history_compactor: CompactionStrategy | None = None,
        history_compress_threshold_tokens: int = -1,
        on_tool_result: Callable[[dict], Any] | None = None,
        on_event: Callable[[dict], Any] | None = None,
        workspace_root: str | None = None,
        inject_project_context: bool = False,
        project_root: str | None = None,
        skills: Iterable[str] | str | None = None,
        redact_tool_result: Callable[[str, Any], Any] | None = None,
        tool_max_concurrency: int = 4,
    ):
        self._llm = llm
        self._tools = tools
        self._budget = budget or Budget()
        self._run_dir = run_dir
        self._system = system
        self._tool_policy = tool_policy
        self._history_compactor = history_compactor
        self._history_compress_threshold_tokens = history_compress_threshold_tokens
        self._on_tool_result = on_tool_result
        self._on_event = on_event
        self._workspace_root = workspace_root
        self._inject_project_context = inject_project_context
        self._project_root = project_root
        self._skills = skills
        self._redact_tool_result = redact_tool_result
        self._tool_max_concurrency = tool_max_concurrency
        self._flow = Flow()

        # Decomposed nodes
        self._llm_node = LLMCallNode()
        self._tool_node = ToolDispatchNode(
            max_concurrency=tool_max_concurrency,
            on_tool_result=on_tool_result,
        )
        self._llm_node - "dispatch_tools" >> self._tool_node
        self._tool_node - "continue" >> self._llm_node

    @classmethod
    def quick(
        cls,
        model: str,
        *,
        adapter: str = "openai",
        tools: str = "builtin",
        workspace: str = ".",
        system: str = "",
        skills: Iterable[str] | str | None = None,
        inject_project_context: bool = False,
        unsafe: bool = False,
        budget: Budget | None = None,
        history_compactor: CompactionStrategy | None = None,
        history_compress_threshold_tokens: int = -1,
        **kwargs: Any,
    ) -> "Agent":
        """One-line Agent factory for quick setup.

        Args:
            model: Model identifier (e.g. "gpt-4o", "claude-sonnet-4-6").
            adapter: "openai" or "anthropic".
            tools: "builtin" to register all built-in tools, or "none".
            workspace: Workspace root directory for file tools.
            system: System prompt.
            skills: Skill name, iterable of skill names, or "all".
            inject_project_context: Load AGENTS.md and .context/ into the system prompt.
            unsafe: If True, use permissive approval (approve all tools).
            budget: Optional budget override.
            history_compactor: Optional custom history compaction strategy.
            history_compress_threshold_tokens: Soft compression threshold. If <= 0,
                defaults to 70% of the LLM context window.
        """
        if adapter == "anthropic":
            from ..adapters.anthropic import AnthropicAdapter
            llm = AnthropicAdapter(model=model)
        else:
            from ..adapters.openai import OpenAIAdapter
            llm = OpenAIAdapter(model=model)

        from ..tools.registry import ToolRegistry, unsafe_defaults
        from ..tools.builtin import BUILTIN_TOOLS

        approval = unsafe_defaults() if unsafe else None
        registry = ToolRegistry(approval_callback=approval)

        if tools == "builtin":
            for t in BUILTIN_TOOLS:
                registry.register(t)

        return cls(
            llm=llm,
            tools=registry,
            budget=budget,
            system=system,
            workspace_root=workspace,
            project_root=workspace,
            skills=skills,
            inject_project_context=inject_project_context,
            history_compactor=history_compactor,
            history_compress_threshold_tokens=history_compress_threshold_tokens,
            **kwargs,
        )

    async def run(self, goal: str) -> RunResult:
        system = self._system
        if self._inject_project_context or self._skills is not None:
            include_all_skills, skill_names = _normalize_skill_selection(self._skills)
            system = build_system_prompt(
                system,
                self._project_root or self._workspace_root or ".",
                include_project_context=self._inject_project_context,
                include_skills=include_all_skills,
                skill_names=skill_names,
            )

        run = Run(
            run_id=_make_ulid(), goal=goal, budget=self._budget,
            created_at=_iso_now(),
            workspace_root=self._workspace_root,
            tool_policy=self._tool_policy,
        )
        store = FileEventStore(self._run_dir, redact_tool_result=self._redact_tool_result)
        if self._on_event is not None:
            store._on_event = self._on_event
        history = HistoryManager(
            max_context_tokens=self._llm.max_context_tokens,
            compactor=self._history_compactor,
            compress_threshold_tokens=self._history_compress_threshold_tokens,
        )

        await store.append(run.run_id, "RUN_CREATED", {"goal": goal})
        history.add("user", goal)

        # Reconfigure ToolDispatchNode with current settings
        self._tool_node = ToolDispatchNode(
            max_concurrency=self._tool_max_concurrency,
            on_tool_result=self._on_tool_result,
        )
        self._llm_node - "dispatch_tools" >> self._tool_node
        self._tool_node - "continue" >> self._llm_node

        shared: dict = {
            "run": run, "llm": self._llm, "tools": self._tools,
            "history": history, "store": store, "system": system,
            "final_answer": "",
        }

        close_reason = "goal_met"
        close_error_class: str | None = None
        try:
            await self._flow.run(shared, self._llm_node)
        except (BudgetExceededError, LoopDetectedError) as e:
            shared["final_answer"] = shared.get("final_answer") or f"[stopped: {e}]"
            close_reason = str(e)
            run.finalize(f"stopped:{type(e).__name__}")
            await store.append(run.run_id, "RUN_STOPPED", {"reason": str(e), "metrics": run.snapshot_metrics()})
        except Exception as e:
            shared["final_answer"] = shared.get("final_answer") or f"[aborted: {e}]"
            close_reason = str(e)
            close_error_class = type(e).__name__
            run.finalize(f"aborted:{type(e).__name__}")
            await store.append_run_aborted(run.run_id, reason=str(e), error_class=type(e).__name__, metrics=run.snapshot_metrics())
        else:
            run.finalize("goal_met")
            await store.append(run.run_id, "RUN_STOPPED", {"reason": "goal_met", "metrics": run.snapshot_metrics()})
        finally:
            try:
                await store.close_open_tool_uses(run.run_id, reason=close_reason, error_class=close_error_class)
            except Exception:
                pass

        return RunResult(
            final_answer=shared.get("final_answer", ""),
            run_id=run.run_id,
            stop_reason=run.stop_reason,
            metrics=run.snapshot_metrics(),
            events_path=str(store.events_path(run.run_id)),
        )

    def run_sync(self, goal: str) -> RunResult:
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            raise RuntimeError(
                "A running event loop was detected (e.g. Jupyter Notebook or an async framework). "
                "Use `await agent.run(goal)` instead of `agent.run_sync(goal)`."
            )
        return asyncio.run(self.run(goal))
