"""Optional critic agent."""

import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Check answer completeness and whether citation markers resolve."""

        if not state.final_answer:
            raise ValueError("Critic requires final_answer")
        cited = {int(value) for value in re.findall(r"\[(\d+)\]", state.final_answer)}
        invalid = sorted(value for value in cited if value < 1 or value > len(state.sources))
        findings = []
        if not cited:
            findings.append("No inline citation markers were found.")
        if invalid:
            findings.append(f"Invalid citation markers: {invalid}.")
        if state.errors:
            findings.append(f"Workflow recorded {len(state.errors)} error(s).")
        state.critic_notes = " ".join(findings) or "Citation and workflow checks passed."
        state.agent_results.append(
            AgentResult(agent=AgentName.CRITIC, content=state.critic_notes)
        )
        state.add_trace_event("critic.completed", {"findings": len(findings)})
        return state
