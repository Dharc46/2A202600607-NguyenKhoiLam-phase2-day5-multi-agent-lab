"""Search client abstraction for ResearcherAgent."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Tavily search client with a deterministic local knowledge fallback."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Return normalized, deduplicated source documents."""

        if self.settings.tavily_api_key:
            results = self._search_tavily(query, max_results)
        else:
            results = self._search_local(query, max_results)
        seen: set[str] = set()
        unique: list[SourceDocument] = []
        for item in results:
            key = item.url or item.title.casefold()
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique[:max_results]

    def _search_tavily(self, query: str, max_results: int) -> list[SourceDocument]:
        payload = json.dumps(
            {
                "api_key": self.settings.tavily_api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "advanced",
            }
        ).encode()
        request = Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.settings.timeout_seconds) as response:
                data = json.loads(response.read().decode())
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Tavily search failed: {exc}") from exc
        return [
            SourceDocument(
                title=item.get("title") or "Untitled source",
                url=item.get("url"),
                snippet=item.get("content") or "",
                metadata={"score": item.get("score"), "provider": "tavily"},
            )
            for item in data.get("results", [])
        ]

    @staticmethod
    def _search_local(query: str, max_results: int) -> list[SourceDocument]:
        references = [
            (
                "Building Effective Agents",
                "https://www.anthropic.com/engineering/building-effective-agents",
                "Agent systems work best when workflows are simple, observable, and matched "
                "to tasks that benefit from delegation and iteration.",
            ),
            (
                "OpenAI Agents orchestration",
                "https://developers.openai.com/api/docs/guides/agents/orchestration",
                "Orchestration coordinates specialized agents, handoffs, tools, and guardrails.",
            ),
            (
                "LangGraph overview",
                "https://docs.langchain.com/oss/python/langgraph/overview",
                "LangGraph supports durable, stateful workflows with conditional routing.",
            ),
            (
                "Microsoft GraphRAG",
                "https://www.microsoft.com/en-us/research/project/graphrag/",
                "GraphRAG extracts entity and relationship graphs to support global and local "
                "reasoning over private text collections.",
            ),
            (
                "NIST AI Risk Management Framework",
                "https://www.nist.gov/itl/ai-risk-management-framework",
                "AI risk management benefits from governance, measurement, monitoring, and "
                "documented controls.",
            ),
        ]
        terms = {word.casefold().strip(".,?!") for word in query.split()}
        ranked = sorted(
            references,
            key=lambda item: sum(term in (item[0] + " " + item[2]).casefold() for term in terms),
            reverse=True,
        )
        return [
            SourceDocument(
                title=title,
                url=url,
                snippet=snippet,
                metadata={"provider": "local", "query": query},
            )
            for title, url, snippet in ranked[:max_results]
        ]
