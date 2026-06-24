"""Supervisor / router."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def run(self, state: ResearchState) -> ResearchState:
        """Choose the next missing workflow stage and record the decision."""

        if state.final_answer:
            route = "done"
        elif state.iteration >= self.settings.max_iterations:
            state.errors.append("Maximum workflow iterations reached; using writer fallback.")
            route = "writer" if not state.final_answer else "done"
        elif not state.research_notes:
            route = "researcher"
        elif not state.analysis_notes:
            route = "analyst"
        else:
            route = "writer"
        state.record_route(route)
        state.add_trace_event(
            "supervisor.route",
            {"next": route, "iteration": state.iteration, "errors": len(state.errors)},
        )
        return state
