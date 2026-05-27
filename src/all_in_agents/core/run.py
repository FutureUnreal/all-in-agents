import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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
    trajectory: list[dict] | None = None


class BudgetExceededError(Exception):
    def __init__(self, dimension: str, current: int, limit: int):
        self.dimension = dimension
        super().__init__(f"Budget exceeded: {dimension}={current}/{limit}")


class LoopDetectedError(Exception):
    def __init__(self, action_sig: str, count: int):
        super().__init__(f"Loop detected: '{action_sig}' repeated {count} times")


class ToolLimitExceededError(Exception):
    def __init__(self, tool_name: str, dimension: str, current: int, limit: int):
        self.tool_name = tool_name
        self.dimension = dimension
        super().__init__(f"Tool limit exceeded: {tool_name} {dimension}={current}/{limit}")


@dataclass
class Budget:
    max_llm_calls: int = 40
    max_tool_calls: int = 80
    max_wall_ms: int = 1_800_000
    max_input_tokens_per_call: int = 24_000
    max_output_tokens_per_call: int = 2_048
    loop_same_action_limit: int = 3


@dataclass
class Run:
    run_id: str
    goal: str
    budget: Budget = field(default_factory=Budget)
    created_at: str = ""
    llm_calls: int = 0
    tool_calls: int = 0
    _start_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    _last_sig: str = field(default="")
    _consecutive_count: int = field(default=0)
    workspace_root: str | None = None
    tool_policy: Any = None
    input_tokens_total: int = 0
    output_tokens_total: int = 0
    tool_mix: dict[str, int] = field(default_factory=dict)
    policy_blocks: int = 0
    compression_count: int = 0
    stop_reason: str = ""
    status: str = RunStatus.SUCCESS.value

    def check_budget(self, action_type: str, action_sig: str = "") -> None:
        elapsed = int(time.time() * 1000) - self._start_ms
        if elapsed >= self.budget.max_wall_ms:
            raise BudgetExceededError("wall_ms", elapsed, self.budget.max_wall_ms)

        if action_type == "llm_call":
            if self.llm_calls >= self.budget.max_llm_calls:
                raise BudgetExceededError("llm_calls", self.llm_calls, self.budget.max_llm_calls)
            self.llm_calls += 1

        elif action_type == "tool_call":
            if self.tool_calls >= self.budget.max_tool_calls:
                raise BudgetExceededError("tool_calls", self.tool_calls, self.budget.max_tool_calls)
            self.tool_calls += 1

            if action_sig:
                sig_key = action_sig[:128]
                if sig_key == self._last_sig:
                    self._consecutive_count += 1
                else:
                    self._last_sig = sig_key
                    self._consecutive_count = 1
                if self._consecutive_count >= self.budget.loop_same_action_limit:
                    raise LoopDetectedError(action_sig, self._consecutive_count)

    def record_llm_usage(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self.input_tokens_total += input_tokens
        self.output_tokens_total += output_tokens
        if input_tokens > 0 and self.budget.max_input_tokens_per_call > 0:
            if input_tokens > self.budget.max_input_tokens_per_call:
                raise BudgetExceededError("input_tokens_per_call", input_tokens, self.budget.max_input_tokens_per_call)
        if output_tokens > 0 and self.budget.max_output_tokens_per_call > 0:
            if output_tokens > self.budget.max_output_tokens_per_call:
                raise BudgetExceededError("output_tokens_per_call", output_tokens, self.budget.max_output_tokens_per_call)

    def record_tool_result(self, tool_name: str, status: str) -> None:
        self.tool_mix[tool_name] = self.tool_mix.get(tool_name, 0) + 1

    def record_policy_block(self, tool_name: str) -> None:
        self.policy_blocks += 1

    def record_compression(self) -> None:
        self.compression_count += 1

    def finalize(self, stop_reason: str, status: str | RunStatus = RunStatus.SUCCESS) -> None:
        self.stop_reason = stop_reason
        self.status = status.value if isinstance(status, RunStatus) else status

    def snapshot_metrics(self) -> dict:
        wall_ms = int(time.time() * 1000) - self._start_ms
        return {
            "stop_reason": self.stop_reason,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "input_tokens_total": self.input_tokens_total,
            "output_tokens_total": self.output_tokens_total,
            "tool_mix": self.tool_mix.copy(),
            "policy_blocks": self.policy_blocks,
            "compression_count": self.compression_count,
            "wall_ms": wall_ms,
            "status": self.status,
        }
