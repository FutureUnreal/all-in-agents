from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Any, Iterable

from ..core.flow import Flow, FlowHooks
from ..core.artifacts import ArtifactContract
from ..core.run import BudgetExceededError, Budget, LoopDetectedError, Run, RunResult, RunStatus, StopReason
from ..adapters.base import GenerationOptions
from ..history.manager import HistoryManager
from ..history.store import FileEventStore
from ..tools.policy import ToolPolicy
from ..history.compactor import CompactionStrategy
from ..agents.harness import build_system_prompt
from ..agents.nodes import LLMCallNode, ReActNode, ToolDispatchNode
from ..utils import make_ulid as _make_ulid, iso_now as _iso_now

if TYPE_CHECKING:
    from ..adapters.base import LLMAdapter, LLMResponse
    from ..tools.registry import ToolRegistry


def _normalize_skill_selection(skills: Iterable[str] | str | None) -> tuple[bool, tuple[str, ...] | None]:
    if skills is None:
        return False, None
    if isinstance(skills, str):
        if skills == "all":
            return True, None
        return False, (skills,)
    return False, tuple(skills)


@dataclass
class AgentConfig:
    """Typed configuration for Agent. All fields have sensible defaults."""

    budget: Budget | None = None
    run_dir: str = "./runs"
    system: str = ""
    tool_policy: ToolPolicy | None = None
    history_compactor: CompactionStrategy | None = None
    history_compress_threshold_tokens: int = -1
    compression_llm: Any = None
    artifact_contract: ArtifactContract | None = None
    on_tool_result: Callable[[dict], Any] | None = None
    on_event: Callable[[dict], Any] | None = None
    flow_hooks: FlowHooks | None = None
    flow_copy_nodes: bool = True
    include_trajectory: bool = False
    workspace_root: str | None = None
    inject_project_context: bool = False
    project_root: str | None = None
    skills: Iterable[str] | str | None = None
    redact_tool_result: Callable[[str, Any], Any] | None = None
    tool_max_concurrency: int = 4


class Agent:
    def __init__(
        self,
        llm: "LLMAdapter",
        tools: "ToolRegistry",
        config: AgentConfig | None = None,
        budget: Budget | None = None,
        run_dir: str = "./runs",
        system: str = "",
        *,
        tool_policy: ToolPolicy | None = None,
        history_compactor: CompactionStrategy | None = None,
        history_compress_threshold_tokens: int = -1,
        compression_llm: "LLMAdapter | None" = None,
        artifact_contract: ArtifactContract | None = None,
        on_tool_result: Callable[[dict], Any] | None = None,
        on_event: Callable[[dict], Any] | None = None,
        flow_hooks: FlowHooks | None = None,
        flow_copy_nodes: bool = True,
        include_trajectory: bool = False,
        workspace_root: str | None = None,
        inject_project_context: bool = False,
        project_root: str | None = None,
        skills: Iterable[str] | str | None = None,
        redact_tool_result: Callable[[str, Any], Any] | None = None,
        tool_max_concurrency: int = 4,
    ):
        if config is not None:
            budget = budget if budget is not None else config.budget
            run_dir = run_dir if run_dir != "./runs" else config.run_dir
            system = system if system != "" else config.system
            tool_policy = tool_policy if tool_policy is not None else config.tool_policy
            history_compactor = history_compactor if history_compactor is not None else config.history_compactor
            history_compress_threshold_tokens = (
                history_compress_threshold_tokens
                if history_compress_threshold_tokens != -1
                else config.history_compress_threshold_tokens
            )
            compression_llm = compression_llm if compression_llm is not None else config.compression_llm
            artifact_contract = artifact_contract if artifact_contract is not None else config.artifact_contract
            on_tool_result = on_tool_result if on_tool_result is not None else config.on_tool_result
            on_event = on_event if on_event is not None else config.on_event
            flow_hooks = flow_hooks if flow_hooks is not None else config.flow_hooks
            flow_copy_nodes = flow_copy_nodes if flow_copy_nodes is not True else config.flow_copy_nodes
            include_trajectory = include_trajectory or config.include_trajectory
            workspace_root = workspace_root if workspace_root is not None else config.workspace_root
            inject_project_context = inject_project_context or config.inject_project_context
            project_root = project_root if project_root is not None else config.project_root
            skills = skills if skills is not None else config.skills
            redact_tool_result = redact_tool_result if redact_tool_result is not None else config.redact_tool_result
            tool_max_concurrency = (
                tool_max_concurrency if tool_max_concurrency != 4 else config.tool_max_concurrency
            )

        self._llm = llm
        self._tools = tools
        self._budget = budget or Budget()
        self._run_dir = run_dir
        self._system = system
        self._tool_policy = tool_policy
        self._history_compactor = history_compactor
        self._history_compress_threshold_tokens = history_compress_threshold_tokens
        self._compression_llm = compression_llm
        self._artifact_contract = artifact_contract
        self._on_tool_result = on_tool_result
        self._on_event = on_event
        self._include_trajectory = include_trajectory
        self._workspace_root = workspace_root
        self._inject_project_context = inject_project_context
        self._project_root = project_root
        self._skills = skills
        self._redact_tool_result = redact_tool_result
        self._tool_max_concurrency = tool_max_concurrency
        self._flow = Flow(hooks=flow_hooks, copy_nodes=flow_copy_nodes)

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
        compression_llm: "LLMAdapter | None" = None,
        artifact_contract: ArtifactContract | None = None,
        api: str = "chat_completions",
        llm_options: GenerationOptions | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        response_format: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
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
            compression_llm: Optional cheaper/specialized LLM used for history compression.
            history_compress_threshold_tokens: Soft compression threshold. If <= 0,
                defaults to 70% of the LLM context window.
            artifact_contract: Optional machine-checkable required output contract.
            api: OpenAI API backend: "chat_completions" or "responses".
            llm_options: Optional provider-neutral generation defaults.
            temperature: Optional sampling temperature passed to the adapter.
            top_p: Optional nucleus sampling value passed to the adapter.
            response_format: Optional structured-output format passed to the adapter.
            reasoning_effort: Optional reasoning-effort hint passed to the adapter.
            model_kwargs: Extra provider-specific request parameters.
        """
        adapter_options = GenerationOptions.from_values(
            llm_options,
            temperature=temperature,
            top_p=top_p,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
            extra=model_kwargs,
        )
        if adapter == "anthropic":
            from ..adapters.anthropic import AnthropicAdapter
            llm = AnthropicAdapter(model=model, options=adapter_options)
        else:
            from ..adapters.openai import OpenAIAdapter
            llm = OpenAIAdapter(model=model, api=api, options=adapter_options)

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
            compression_llm=compression_llm,
            artifact_contract=artifact_contract,
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
            "compression_llm": self._compression_llm or self._llm,
            "history": history, "store": store, "system": system,
            "final_answer": "",
        }

        close_reason = "goal_met"
        close_error_class: str | None = None
        artifact_validation: dict | None = None
        try:
            await self._flow.run(shared, self._llm_node)
        except (BudgetExceededError, LoopDetectedError) as e:
            shared["final_answer"] = shared.get("final_answer") or f"[stopped: {e}]"
            close_reason = str(e)
            if isinstance(e, BudgetExceededError):
                run.finalize(StopReason.BUDGET_EXHAUSTED.value, RunStatus.BUDGET_EXHAUSTED)
            else:
                run.finalize(StopReason.LOOP_DETECTED.value, RunStatus.INCOMPLETE)
            await store.append(run.run_id, "RUN_STOPPED", {"reason": str(e), "metrics": run.snapshot_metrics()})
        except Exception as e:
            shared["final_answer"] = shared.get("final_answer") or f"[aborted: {e}]"
            close_reason = str(e)
            close_error_class = type(e).__name__
            run.finalize(f"{StopReason.ABORTED.value}:{type(e).__name__}", RunStatus.ERROR)
            await store.append_run_aborted(run.run_id, reason=str(e), error_class=type(e).__name__, metrics=run.snapshot_metrics())
        else:
            run.finalize(StopReason.GOAL_MET.value, RunStatus.SUCCESS)
            if self._artifact_contract is not None:
                result = self._artifact_contract.validate(self._workspace_root or ".")
                artifact_validation = result.to_dict()
                await store.append(run.run_id, "ARTIFACT_VALIDATION", artifact_validation)
                if not result.ok:
                    reason = StopReason.ARTIFACT_MISSING.value
                    if any("schema" in err.lower() or "json" in err.lower() for err in result.errors):
                        reason = StopReason.VALIDATION_FAILED.value
                    run.finalize(reason, RunStatus.INCOMPLETE)
            await store.append(run.run_id, "RUN_STOPPED", {"reason": run.stop_reason, "metrics": run.snapshot_metrics()})
        finally:
            try:
                await store.close_open_tool_uses(run.run_id, reason=close_reason, error_class=close_error_class)
            except Exception:
                pass

        trajectory = store.build_trajectory(run.run_id) if self._include_trajectory else None

        return RunResult(
            final_answer=shared.get("final_answer", ""),
            run_id=run.run_id,
            stop_reason=run.stop_reason,
            metrics=run.snapshot_metrics(),
            events_path=str(store.events_path(run.run_id)),
            status=run.status,
            artifact_validation=artifact_validation,
            trajectory=trajectory,
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
