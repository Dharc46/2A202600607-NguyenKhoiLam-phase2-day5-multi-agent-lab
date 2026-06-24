"""Researcher agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self, search_client: SearchClient | None = None) -> None:
        self.search_client = search_client or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Search, normalize citations, and populate concise source-grounded notes."""

        sources = self.search_client.search(
            state.request.query,
            max_results=state.request.max_sources,
        )
        if not sources:
            raise RuntimeError("Search returned no sources")
        state.sources = sources
        notes = [
            f"[{index}] {source.title}: {source.snippet.strip()}"
            for index, source in enumerate(sources, start=1)
        ]
        state.research_notes = "\n".join(notes)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=state.research_notes,
                metadata={"source_count": len(sources)},
            )
        )
        state.add_trace_event("researcher.completed", {"source_count": len(sources)})
        return state
