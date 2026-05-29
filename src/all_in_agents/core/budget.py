from __future__ import annotations

import time
from dataclasses import dataclass, field


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
    max_input_tokens_per_call: int = 0
    max_output_tokens_per_call: int = 2_048
    loop_same_action_limit: int = 3


@dataclass
class BudgetLedger:
    budget: Budget = field(default_factory=Budget)
    parent: "BudgetLedger | None" = None
    start_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    llm_calls: int = 0
    tool_calls: int = 0
    input_tokens_total: int = 0
    output_tokens_total: int = 0
    tool_mix: dict[str, int] = field(default_factory=dict)
    policy_blocks: int = 0
    compression_count: int = 0
    _last_sig: str = ""
    _consecutive_count: int = 0

    def child(self, budget: Budget | None = None) -> "BudgetLedger":
        return BudgetLedger(budget=budget or self.budget, parent=self)

    def check_budget(self, action_type: str, action_sig: str = "") -> None:
        self._consume_local(action_type, action_sig)
        if self.parent is not None:
            self.parent.check_budget(action_type, action_sig)

    def _consume_local(self, action_type: str, action_sig: str = "") -> None:
        elapsed = int(time.time() * 1000) - self.start_ms
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
        self._record_llm_usage_local(input_tokens, output_tokens)
        if self.parent is not None:
            self.parent.record_llm_usage(input_tokens, output_tokens)

    def _record_llm_usage_local(self, input_tokens: int, output_tokens: int) -> None:
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
        if self.parent is not None:
            self.parent.record_tool_result(tool_name, status)

    def record_policy_block(self, tool_name: str) -> None:
        self.policy_blocks += 1
        if self.parent is not None:
            self.parent.record_policy_block(tool_name)

    def record_compression(self) -> None:
        self.compression_count += 1
        if self.parent is not None:
            self.parent.record_compression()

    def snapshot_metrics(self, stop_reason: str = "", status: str = "") -> dict:
        wall_ms = int(time.time() * 1000) - self.start_ms
        return {
            "stop_reason": stop_reason,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "input_tokens_total": self.input_tokens_total,
            "output_tokens_total": self.output_tokens_total,
            "tool_mix": self.tool_mix.copy(),
            "policy_blocks": self.policy_blocks,
            "compression_count": self.compression_count,
            "wall_ms": wall_ms,
            "status": status,
        }

    def to_checkpoint(self) -> dict:
        return {
            "elapsed_ms": int(time.time() * 1000) - self.start_ms,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "input_tokens_total": self.input_tokens_total,
            "output_tokens_total": self.output_tokens_total,
            "tool_mix": self.tool_mix.copy(),
            "policy_blocks": self.policy_blocks,
            "compression_count": self.compression_count,
            "last_sig": self._last_sig,
            "consecutive_count": self._consecutive_count,
        }

    def restore_checkpoint(self, data: dict) -> None:
        elapsed_ms = int(data.get("elapsed_ms", 0) or 0)
        self.start_ms = int(time.time() * 1000) - elapsed_ms
        self.llm_calls = int(data.get("llm_calls", 0) or 0)
        self.tool_calls = int(data.get("tool_calls", 0) or 0)
        self.input_tokens_total = int(data.get("input_tokens_total", 0) or 0)
        self.output_tokens_total = int(data.get("output_tokens_total", 0) or 0)
        self.tool_mix = dict(data.get("tool_mix") or {})
        self.policy_blocks = int(data.get("policy_blocks", 0) or 0)
        self.compression_count = int(data.get("compression_count", 0) or 0)
        self._last_sig = data.get("last_sig", "")
        self._consecutive_count = int(data.get("consecutive_count", 0) or 0)
