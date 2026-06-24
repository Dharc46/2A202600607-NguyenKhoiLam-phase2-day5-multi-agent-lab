"""Writer agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Synthesize a readable answer with a verifiable source list."""

        evidence = state.analysis_notes or state.research_notes
        if not evidence:
            evidence = "No external evidence was available; state this limitation."
        response = self.llm_client.complete(
            "You are a research writer. Answer for the requested audience, distinguish "
            "evidence from inference, use [n] citations, and do not invent sources.",
            f"Question: {state.request.query}\nAudience: {state.request.audience}\n\n"
            f"Evidence:\n{evidence}",
        )
        source_lines = [
            f"[{index}] {source.title}"
            + (f" — {source.url}" if source.url else "")
            for index, source in enumerate(state.sources, start=1)
        ]
        source_section = "\n".join(source_lines) if source_lines else "No external sources."
        state.final_answer = f"{response.content.strip()}\n\n## Sources\n{source_section}"
        state.add_usage(response.input_tokens, response.output_tokens, response.cost_usd)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=state.final_answer,
                metadata={"source_count": len(state.sources)},
            )
        )
        state.add_trace_event("writer.completed", {"output_chars": len(state.final_answer)})
        return state
