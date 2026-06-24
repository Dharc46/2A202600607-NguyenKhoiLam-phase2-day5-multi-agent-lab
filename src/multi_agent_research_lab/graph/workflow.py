"""LangGraph workflow with a dependency-free fallback runner."""

from importlib import import_module
from time import perf_counter
from typing import Any, Protocol

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import export_trace, trace_span


class GraphRunner(Protocol):
    def invoke(
        self,
        state: ResearchState,
        config: dict[str, Any] | None = None,
    ) -> ResearchState | dict[str, Any]: ...


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        supervisor: SupervisorAgent | None = None,
        researcher: ResearcherAgent | None = None,
        analyst: AnalystAgent | None = None,
        writer: WriterAgent | None = None,
        critic: CriticAgent | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.supervisor = supervisor or SupervisorAgent(self.settings)
        self.researcher = researcher or ResearcherAgent()
        self.analyst = analyst or AnalystAgent()
        self.writer = writer or WriterAgent()
        self.critic = critic or CriticAgent()
        self._compiled: GraphRunner | None = None

    def build(self) -> GraphRunner:
        """Create and compile the LangGraph graph when the optional package is installed."""

        if self._compiled is not None:
            return self._compiled
        try:
            graph_module = import_module("langgraph.graph")
        except ImportError:
            self._compiled = _FallbackGraph(self)
            return self._compiled

        END = graph_module.END
        START = graph_module.START
        StateGraph = graph_module.StateGraph
        graph = StateGraph(ResearchState)
        graph.add_node("supervisor", self.supervisor.run)
        graph.add_node("researcher", self.researcher.run)
        graph.add_node("analyst", self.analyst.run)
        graph.add_node("writer", self.writer.run)
        graph.add_node("critic", self.critic.run)
        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            lambda state: state.route_history[-1],
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "done": END,
            },
        )
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "critic")
        graph.add_edge("critic", END)
        self._compiled = graph.compile()
        return self._compiled

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph with tracing, bounded retries, and output validation."""

        started = perf_counter()
        graph = self.build()
        final_state: ResearchState | None = None
        try:
            result = graph.invoke(
                state,
                config={"recursion_limit": self.settings.max_iterations * 2 + 4},
            )
            final_state = (
                result
                if isinstance(result, ResearchState)
                else ResearchState.model_validate(result)
            )
        except Exception as exc:
            state.errors.append(f"{type(exc).__name__}: {exc}")
            state.add_trace_event("workflow.failed", {"error": str(exc)})
            raise AgentExecutionError(f"Multi-agent workflow failed: {exc}") from exc
        finally:
            traced_state = final_state or state
            traced_state.add_trace_event(
                "workflow.duration",
                {"seconds": perf_counter() - started},
            )
            export_trace(traced_state.trace, self.settings.trace_path)
        assert final_state is not None
        if not final_state.final_answer:
            raise AgentExecutionError("Workflow completed without a final answer")
        return final_state

    def _run_fallback(self, state: ResearchState) -> ResearchState:
        deadline = perf_counter() + self.settings.timeout_seconds
        agents = {
            "researcher": self.researcher,
            "analyst": self.analyst,
            "writer": self.writer,
        }
        while not state.final_answer:
            if perf_counter() >= deadline:
                raise TimeoutError(f"Workflow exceeded {self.settings.timeout_seconds} seconds")
            self.supervisor.run(state)
            route = state.route_history[-1]
            if route == "done":
                break
            agent = agents[route]
            with trace_span(f"agent.{route}", {"iteration": state.iteration}) as span:
                last_error: Exception | None = None
                for attempt in range(1, 3):
                    try:
                        agent.run(state)
                        span["attributes"]["attempt"] = attempt
                        break
                    except Exception as exc:
                        last_error = exc
                        state.errors.append(f"{route} attempt {attempt}: {exc}")
                else:
                    if route == "writer":
                        raise AgentExecutionError("Writer failed after retries") from last_error
                    # Continue through the pipeline with an explicit degraded-mode note.
                    fallback = f"{route.title()} unavailable: {last_error}"
                    if route == "researcher":
                        state.research_notes = fallback
                    else:
                        state.analysis_notes = fallback
                state.trace.append(span)
        self.critic.run(state)
        return state


class _FallbackGraph:
    """Small adapter matching LangGraph's invoke interface for minimal installs."""

    def __init__(self, workflow: MultiAgentWorkflow) -> None:
        self.workflow = workflow

    def invoke(
        self,
        state: ResearchState,
        config: dict[str, Any] | None = None,
    ) -> ResearchState:
        del config
        return self.workflow._run_fallback(state)
