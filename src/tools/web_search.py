from langchain_core.tools import tool

from config.settings import settings


def create_web_search_tool():
    """Create a DuckDuckGo web search tool.

    Returns a LangChain tool that searches the web using DuckDuckGo.
    This is a free, no-API-key-needed search engine.

    Returns:
        A LangChain @tool decorated function.
    """

    max_results = settings.WEB_SEARCH_MAX_RESULTS

    @tool
    def web_search(query: str) -> str:
        """Search the web for current information on a topic.

        Use this tool to find the latest information, verify facts,
        get up-to-date documentation, or find recent developments.
        Returns search results with titles, URLs, and content snippets.

        Args:
            query: The search query string.
        """
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))

            if not results:
                return "No web search results found for this query."

            formatted = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "N/A")
                url = r.get("href", "N/A")
                snippet = r.get("body", "N/A")
                formatted.append(
                    f"**Result {i}: {title}**\n"
                    f"URL: {url}\n"
                    f"{snippet}"
                )

            return "\n\n---\n\n".join(formatted)

        except Exception as e:
            return (
                f"Web search encountered an error: {str(e)}. "
                "Proceeding without web results — rely on your existing knowledge."
            )

    return web_search
