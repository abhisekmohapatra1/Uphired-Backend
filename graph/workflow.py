from langgraph.graph import StateGraph, add_messages
from langgraph.constants import END
from langgraph.checkpoint.memory import InMemorySaver
from graph.state import AgentState
from agents.planner_agent import planner_node
from agents.browser_agent import browser_node
from agents.extractor_agent import extractor_node
from agents.ranker_agent import ranker_node
from agents.summary_agent import summary_node
from loguru import logger


def should_retry_browser(state: AgentState) -> str:
    if not state["raw_html_pages"] and state["retry_count"] < state["max_retries"]:
        return "retry_browser"
    return "extractor"


def should_broaden_search(state: AgentState) -> str:
    if not state["extracted_jobs"]:
        if not state["search_broadened"]:
            return "broaden_search"
        else:
            return "summary"
    return "ranker"


def make_browser_node(browser_tools=None):
    """LangGraph node factory with injectable tool implementation (real or fake)."""
    async def _node(state: AgentState) -> dict:
        return await browser_node(state, browser_tools=browser_tools)
    return _node


def build_graph(checkpointer=None, browser_tools=None):
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("browser", make_browser_node(browser_tools))
    graph.add_node("extractor", extractor_node)
    graph.add_node("ranker", ranker_node)
    graph.add_node("summary", summary_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "browser")

    graph.add_conditional_edges(
        "browser",
        should_retry_browser,
        {
            "retry_browser": "browser",
            "extractor": "extractor",
        }
    )

    graph.add_conditional_edges(
        "extractor",
        should_broaden_search,
        {
            "broaden_search": "planner",
            "ranker": "ranker",
            "summary": "summary",
        }
    )

    graph.add_edge("ranker", "summary")
    graph.add_edge("summary", END)

    return graph.compile(checkpointer=checkpointer)


# Module-level checkpointer — created once, reused across requests
# InMemorySaver works perfectly for Phase 1; upgrade to SQLite/Postgres in Phase 5
_checkpointer = InMemorySaver()


async def get_graph():
    """
    LangGraph 1.x — AsyncSqliteSaver requires async context manager usage
    which doesn't work well with FastAPI background tasks.
    InMemorySaver is the correct approach for Phase 1.
    Workflows persist in memory for the server lifetime.
    """
    logger.info("Using InMemorySaver checkpointer")
    return build_graph(checkpointer=_checkpointer)