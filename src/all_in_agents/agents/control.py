from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..adapters.base import LLMResponse
from ..core.run import RunStatus, StopReason


@dataclass
class AgentTurn:
    """A single completed model turn before the agent decides the next action."""

    run_id: str
    goal: str
    step_index: int
    response: LLMResponse
    messages: list[dict]
    tools: list[dict]
    system: str
    state: dict[str, Any]


@dataclass
class AgentTurnDecision:
    """Decision returned by an Agent turn gate.

    Use an async callback when the caller needs to pause for human approval or
    external policy checks. Returning None is equivalent to continue_().
    """

    action: str = "continue"
    response: LLMResponse | None = None
    final_answer: str | None = None
    stop_reason: str = "user_stopped"
    status: str = RunStatus.INTERRUPTED.value

    @classmethod
    def continue_(cls, response: LLMResponse | None = None) -> "AgentTurnDecision":
        return cls(action="continue", response=response)

    @classmethod
    def replace(cls, response: LLMResponse) -> "AgentTurnDecision":
        return cls(action="continue", response=response)

    @classmethod
    def stop(
        cls,
        final_answer: str | None = None,
        *,
        stop_reason: str = "user_stopped",
        status: str = RunStatus.INTERRUPTED.value,
    ) -> "AgentTurnDecision":
        return cls(
            action="stop",
            final_answer=final_answer,
            stop_reason=stop_reason,
            status=status,
        )

    @classmethod
    def accept(cls, final_answer: str | None = None) -> "AgentTurnDecision":
        return cls.stop(
            final_answer=final_answer,
            stop_reason=StopReason.GOAL_MET.value,
            status=RunStatus.SUCCESS.value,
        )

    @classmethod
    def abort(
        cls,
        final_answer: str | None = None,
        *,
        stop_reason: str = StopReason.ABORTED.value,
    ) -> "AgentTurnDecision":
        return cls.stop(
            final_answer=final_answer,
            stop_reason=stop_reason,
            status=RunStatus.ERROR.value,
        )


def normalize_turn_decision(value: Any) -> AgentTurnDecision:
    if value is None:
        return AgentTurnDecision.continue_()
    if isinstance(value, AgentTurnDecision):
        return value
    if isinstance(value, LLMResponse):
        return AgentTurnDecision.replace(value)
    if isinstance(value, dict):
        response = value.get("response")
        if response is not None and not isinstance(response, LLMResponse):
            raise TypeError("Agent turn decision 'response' must be an LLMResponse")
        return AgentTurnDecision(
            action=value.get("action", "continue"),
            response=response,
            final_answer=value.get("final_answer"),
            stop_reason=value.get("stop_reason", "user_stopped"),
            status=value.get("status", RunStatus.INTERRUPTED.value),
        )
    raise TypeError(
        "Agent turn gate must return None, LLMResponse, AgentTurnDecision, or dict"
    )
