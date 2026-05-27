from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .checkpoint import to_jsonable
from .budget import (
    Budget,
    BudgetLedger,
)
from .trace import RunTrace


class RunStatus(str, Enum):
    SUCCESS = "success"
    INCOMPLETE = "incomplete"
    ERROR = "error"
    BUDGET_EXHAUSTED = "budget_exhausted"
    INTERRUPTED = "interrupted"


class StopReason(str, Enum):
    GOAL_MET = "goal_met"
    ARTIFACT_MISSING = "artifact_missing"
    VALIDATION_FAILED = "validation_failed"
    MODEL_UNAVAILABLE = "model_unavailable"
    TOOL_FAILED = "tool_failed"
    POLICY_BLOCKED = "policy_blocked"
    BUDGET_EXHAUSTED = "budget_exhausted"
    LOOP_DETECTED = "loop_detected"
    ABORTED = "aborted"


@dataclass
class RunResult:
    """Typed return value from Agent.run()."""

    final_answer: str
    run_id: str
    stop_reason: str
    metrics: dict
    events_path: str
    status: str = RunStatus.SUCCESS.value
    artifact_validation: dict | None = None
    trace: RunTrace | None = None
    checkpoint_path: str = ""

    @property
    def trajectory(self) -> list[dict] | None:
        return self.trace.trajectory if self.trace is not None else None


@dataclass
class Run:
    run_id: str
    goal: str
    budget: Budget = field(default_factory=Budget)
    created_at: str = ""
    ledger: BudgetLedger | None = None
    parent_run_id: str | None = None
    workspace_root: str | None = None
    tool_policy: Any = None
    stop_reason: str = ""
    status: str = RunStatus.SUCCESS.value

    def __post_init__(self) -> None:
        if self.ledger is None:
            self.ledger = BudgetLedger(self.budget)

    @property
    def llm_calls(self) -> int:
        return self.ledger.llm_calls

    @property
    def tool_calls(self) -> int:
        return self.ledger.tool_calls

    @property
    def input_tokens_total(self) -> int:
        return self.ledger.input_tokens_total

    @property
    def output_tokens_total(self) -> int:
        return self.ledger.output_tokens_total

    @property
    def tool_mix(self) -> dict[str, int]:
        return self.ledger.tool_mix

    @property
    def policy_blocks(self) -> int:
        return self.ledger.policy_blocks

    @property
    def compression_count(self) -> int:
        return self.ledger.compression_count

    def spawn_child(self, run_id: str, goal: str, budget: Budget | None = None, **kwargs: Any) -> "Run":
        return Run(
            run_id=run_id,
            goal=goal,
            budget=budget or self.budget,
            ledger=self.ledger.child(budget),
            parent_run_id=self.run_id,
            workspace_root=kwargs.get("workspace_root", self.workspace_root),
            tool_policy=kwargs.get("tool_policy", self.tool_policy),
            created_at=kwargs.get("created_at", ""),
        )

    def to_checkpoint(self) -> dict:
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "budget": to_jsonable(self.budget),
            "created_at": self.created_at,
            "parent_run_id": self.parent_run_id,
            "workspace_root": self.workspace_root,
            "stop_reason": self.stop_reason,
            "status": self.status,
            "ledger": self.ledger.to_checkpoint(),
        }

    def restore_checkpoint(self, data: dict) -> None:
        self.run_id = data.get("run_id", self.run_id)
        self.goal = data.get("goal", self.goal)
        self.created_at = data.get("created_at", self.created_at)
        self.parent_run_id = data.get("parent_run_id", self.parent_run_id)
        self.workspace_root = data.get("workspace_root", self.workspace_root)
        self.stop_reason = data.get("stop_reason", self.stop_reason)
        self.status = data.get("status", self.status)
        if self.ledger is None:
            self.ledger = BudgetLedger(self.budget)
        self.ledger.restore_checkpoint(data.get("ledger") or {})

    def check_budget(self, action_type: str, action_sig: str = "") -> None:
        self.ledger.check_budget(action_type, action_sig)

    def record_llm_usage(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self.ledger.record_llm_usage(input_tokens, output_tokens)

    def record_tool_result(self, tool_name: str, status: str) -> None:
        self.ledger.record_tool_result(tool_name, status)

    def record_policy_block(self, tool_name: str) -> None:
        self.ledger.record_policy_block(tool_name)

    def record_compression(self) -> None:
        self.ledger.record_compression()

    def finalize(self, stop_reason: str, status: str | RunStatus = RunStatus.SUCCESS) -> None:
        self.stop_reason = stop_reason
        self.status = status.value if isinstance(status, RunStatus) else status

    def snapshot_metrics(self) -> dict:
        metrics = self.ledger.snapshot_metrics(self.stop_reason, self.status)
        if self.parent_run_id:
            metrics["parent_run_id"] = self.parent_run_id
        return metrics
