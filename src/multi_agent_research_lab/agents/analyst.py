"""Analyst agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Extract claims, agreements, caveats, and evidence gaps."""

        if not state.research_notes:
            raise ValueError("Analyst requires research_notes")
        response = self.llm_client.complete(
            "You are an evidence analyst. Extract key claims, compare sources, preserve "
            "the [n] citation markers, and explicitly flag weak or missing evidence.",
            f"Question: {state.request.query}\n\nResearch notes:\n{state.research_notes}",
        )
        state.analysis_notes = response.content
        state.add_usage(response.input_tokens, response.output_tokens, response.cost_usd)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                },
            )
        )
        state.add_trace_event("analyst.completed", {"output_chars": len(response.content)})
        return state
