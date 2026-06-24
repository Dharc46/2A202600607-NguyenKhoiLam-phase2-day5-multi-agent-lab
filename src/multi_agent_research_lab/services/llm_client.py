"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """OpenAI client with a deterministic offline fallback."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: Any | None = None
        if self.settings.openai_api_key:
            try:
                openai_module = import_module("openai")
            except ImportError as exc:
                raise AgentExecutionError(
                    "OPENAI_API_KEY is set but the OpenAI package is not installed; "
                    'install the project with the "llm" extra.'
                ) from exc
            OpenAI = openai_module.OpenAI
            self._client = OpenAI(
                api_key=self.settings.openai_api_key,
                timeout=self.settings.timeout_seconds,
            )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=4), reraise=True)
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a completion and normalized usage metadata."""

        if self._client is None:
            return self._offline_complete(system_prompt, user_prompt)

        response = self._client.chat.completions.create(
            model=self.settings.openai_model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None
        return LLMResponse(
            content=content.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self._estimate_cost(input_tokens, output_tokens),
        )

    @staticmethod
    def _estimate_cost(input_tokens: int | None, output_tokens: int | None) -> float | None:
        if input_tokens is None and output_tokens is None:
            return None
        # Configurable providers expose different prices. These conservative defaults
        # match a low-cost model and are clearly estimates, not billing data.
        return ((input_tokens or 0) * 0.15 + (output_tokens or 0) * 0.60) / 1_000_000

    @staticmethod
    def _offline_complete(system_prompt: str, user_prompt: str) -> LLMResponse:
        words = user_prompt.split()
        excerpt = " ".join(words[:220])
        content = (
            f"{excerpt}\n\n"
            "This response was produced by the deterministic offline provider. "
            "Configure OPENAI_API_KEY for model-generated synthesis."
        )
        return LLMResponse(
            content=content,
            input_tokens=len(system_prompt.split()) + len(words),
            output_tokens=len(content.split()),
            cost_usd=0.0,
        )
