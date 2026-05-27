from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..utils import make_ulid as _make_ulid, iso_now as _iso_now

if TYPE_CHECKING:
    from ..adapters.base import LLMAdapter
    from ..tools.registry import ToolRegistry


@dataclass
class Step:
    name: str
    goal: str
    depends_on: tuple[str, ...] = ()
    config: Any = None


@dataclass
class StepResult:
    name: str
    status: str
    run_result: Any = None


@dataclass
class WorkflowResult:
    workflow_id: str
    steps: dict[str, StepResult]
    status: str
    checkpoint_path: str


def _toposort(steps: list[Step]) -> list[Step]:
    graph: dict[str, set[str]] = {s.name: set(s.depends_on) for s in steps}
    by_name = {s.name: s for s in steps}
    ordered: list[Step] = []
    visited: set[str] = set()
    temp: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in temp:
            raise ValueError(f"Cycle detected involving step '{name}'")
        temp.add(name)
        for dep in graph.get(name, set()):
            if dep not in by_name:
                raise ValueError(f"Step '{name}' depends on unknown step '{dep}'")
            visit(dep)
        temp.remove(name)
        visited.add(name)
        ordered.append(by_name[name])

    for s in steps:
        visit(s.name)
    return ordered


class Workflow:
    def __init__(
        self,
        llm: "LLMAdapter",
        tools: "ToolRegistry",
        *,
        config: Any = None,
        checkpoint_dir: str = "./checkpoints",
    ):
        self._llm = llm
        self._tools = tools
        self._config = config
        self._checkpoint_dir = checkpoint_dir
        self._steps: list[Step] = []

    def step(
        self,
        name: str,
        goal: str,
        *,
        depends_on: list[str] | None = None,
        config: Any = None,
    ) -> "Workflow":
        self._steps.append(Step(
            name=name,
            goal=goal,
            depends_on=tuple(depends_on) if depends_on else (),
            config=config,
        ))
        return self

    async def run(self, *, resume_from: str | None = None) -> WorkflowResult:
        from ..agents.base import Agent

        workflow_id = resume_from or _make_ulid()
        cp_dir = Path(self._checkpoint_dir) / workflow_id
        cp_dir.mkdir(parents=True, exist_ok=True)
        cp_path = cp_dir / "checkpoint.json"

        completed: dict[str, dict] = {}
        if resume_from and cp_path.exists():
            data = json.loads(cp_path.read_text(encoding="utf-8"))
            completed = data.get("completed", {})

        ordered = _toposort(self._steps)
        results: dict[str, StepResult] = {}

        for step_name, info in completed.items():
            results[step_name] = StepResult(
                name=step_name,
                status=info.get("status", "success"),
                run_result=None,
            )

        for step in ordered:
            previous = completed.get(step.name) or {}
            if previous.get("status") == "success":
                continue

            for dep in step.depends_on:
                dep_status = completed.get(dep, {}).get("status")
                if dep_status != "success":
                    results[step.name] = StepResult(name=step.name, status="skipped")
                    self._save_checkpoint(cp_path, workflow_id, completed, step.name)
                    return WorkflowResult(
                        workflow_id=workflow_id,
                        steps=results,
                        status="partial",
                        checkpoint_path=str(cp_path),
                    )

            context_parts: list[str] = []
            for dep in step.depends_on:
                dep_answer = completed.get(dep, {}).get("final_answer", "")
                if dep_answer:
                    context_parts.append(f"### Step: {dep}\nResult: {dep_answer}")

            goal = step.goal
            if context_parts:
                goal = "## Prior Step Results\n" + "\n".join(context_parts) + "\n\n## Current Task\n" + goal

            step_config = step.config or self._config
            agent = Agent(self._llm, self._tools, config=step_config)

            try:
                run_result = await agent.run(
                    goal,
                    checkpoint=True,
                    resume_from=previous.get("run_id") or None,
                )
                status = run_result.status
                if status != "success":
                    status = "error"
            except Exception:
                run_result = None
                status = "error"

            results[step.name] = StepResult(name=step.name, status=status, run_result=run_result)

            completed[step.name] = {
                "status": status,
                "final_answer": run_result.final_answer if run_result else "",
                "run_id": run_result.run_id if run_result else "",
                "events_path": run_result.events_path if run_result else "",
                "checkpoint_path": run_result.checkpoint_path if run_result else "",
            }
            self._save_checkpoint(cp_path, workflow_id, completed, None)

            if status != "success":
                return WorkflowResult(
                    workflow_id=workflow_id,
                    steps=results,
                    status="partial",
                    checkpoint_path=str(cp_path),
                )

        final_status = "success" if all(r.status == "success" for r in results.values()) else "failed"
        return WorkflowResult(
            workflow_id=workflow_id,
            steps=results,
            status=final_status,
            checkpoint_path=str(cp_path),
        )

    def run_sync(self, *, resume_from: str | None = None) -> WorkflowResult:
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            raise RuntimeError(
                "A running event loop was detected. Use `await workflow.run()` instead."
            )
        return asyncio.run(self.run(resume_from=resume_from))

    def _save_checkpoint(
        self,
        cp_path: Path,
        workflow_id: str,
        completed: dict[str, dict],
        failed_step: str | None,
    ) -> None:
        pending = [s.name for s in self._steps if completed.get(s.name, {}).get("status") != "success"]
        data = {
            "workflow_id": workflow_id,
            "created_at": _iso_now(),
            "steps_definition": [
                {"name": s.name, "goal": s.goal, "depends_on": list(s.depends_on)}
                for s in self._steps
            ],
            "completed": completed,
            "failed_step": failed_step,
            "pending": pending,
        }
        tmp = cp_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(cp_path)
