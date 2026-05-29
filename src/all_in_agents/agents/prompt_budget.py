from __future__ import annotations

from dataclasses import dataclass

from ..core.tokens import estimate_data_tokens, estimate_text_tokens


@dataclass(frozen=True)
class PromptBudget:
    model_context_tokens: int
    prompt_cap_tokens: int
    output_reserve_tokens: int
    static_tokens: int
    history_tokens: int

    def to_dict(self) -> dict:
        return {
            "model_context_tokens": self.model_context_tokens,
            "prompt_cap_tokens": self.prompt_cap_tokens,
            "output_reserve_tokens": self.output_reserve_tokens,
            "static_tokens": self.static_tokens,
            "history_tokens": self.history_tokens,
        }


class PromptBudgeter:
    """Allocate per-call history budget after static prompt overhead."""

    def __init__(self, static_padding_tokens: int = 128):
        self.static_padding_tokens = static_padding_tokens

    def allocate(
        self,
        *,
        model_context_tokens: int,
        max_output_tokens: int,
        max_input_tokens_per_call: int = 0,
        system: str = "",
        tools: list[dict] | None = None,
    ) -> PromptBudget:
        raw_output_reserve = max(0, max_output_tokens)
        output_reserve = raw_output_reserve
        if model_context_tokens > 0 and output_reserve >= model_context_tokens:
            output_reserve = model_context_tokens // 2
        model_prompt_cap = max(0, model_context_tokens - output_reserve)
        prompt_cap = model_prompt_cap
        if max_input_tokens_per_call > 0:
            prompt_cap = min(prompt_cap, max_input_tokens_per_call)

        static_tokens = (
            estimate_text_tokens(system)
            + estimate_data_tokens(tools or [])
            + self.static_padding_tokens
        )
        history_tokens = max(0, prompt_cap - static_tokens)
        return PromptBudget(
            model_context_tokens=model_context_tokens,
            prompt_cap_tokens=prompt_cap,
            output_reserve_tokens=output_reserve,
            static_tokens=static_tokens,
            history_tokens=history_tokens,
        )
