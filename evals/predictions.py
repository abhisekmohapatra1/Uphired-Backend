"""Prediction functions run by the eval harness against the golden datasets.

`prompt_predict`   — invokes the agent's real system prompt against the LLM.
`graph_predict`    — runs the full LangGraph workflow with a FakeBrowser so
                     agent orchestration + tool dispatch are evaluated hermetically.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from loguru import logger

from agents.llm_client import OpenRouterLLM
from config import settings
from evals.fixtures import FakeBrowser

# agent → (model, temperature) mirroring each agent module's llm config
PROMPT_LLMS: dict[str, OpenRouterLLM] = {
    "planner":         OpenRouterLLM(model=settings.PLANNER_MODEL, temperature=0.1),
    "planner_broaden": OpenRouterLLM(model=settings.PLANNER_MODEL, temperature=0.1),
    "extractor":       OpenRouterLLM(model=settings.EXTRACTOR_MODEL, temperature=0),
    "ranker_relevance": OpenRouterLLM(model=settings.PLANNER_MODEL, temperature=0),
    "ranker_score":     OpenRouterLLM(model=settings.PLANNER_MODEL, temperature=0),
    "summary":          OpenRouterLLM(model=settings.SUMMARY_MODEL, temperature=0.3),
    "resume_parser":    OpenRouterLLM(model=settings.EXTRACTOR_MODEL, temperature=0),
}
DEFAULT_LLM = OpenRouterLLM(model=settings.EXTRACTOR_MODEL, temperature=0)


async def prompt_predict(inputs: dict) -> dict:
    """Calls the system prompt on the LLM and returns the raw model text."""
    agent = inputs.get("agent", "")
    llm = PROMPT_LLMS.get(agent, DEFAULT_LLM)
    try:
        response = await llm.ainvoke([
            SystemMessage(content=inputs["system_prompt"]),
            HumanMessage(content=inputs["human_text"]),
        ])
        return {"content": response.content}
    except Exception as e:
        logger.error(f"[eval] prompt_predict({agent}) failed: {e}")
        return {"content": "", "error": str(e)}


def _base_state(inputs: dict) -> dict:
    return {
        "user_query": inputs.get("user_query", ""),
        "user_skills": inputs.get("skills", []),
        "experience_years": inputs.get("experience_years", 0),
        "preferred_sites": inputs.get("sites", []),
        "market": inputs.get("market", "global"),
        "messages": [],
        "current_step": "planner",
        "retry_count": 0,
        "max_retries": 2,
        "search_broadened": False,
        "search_plan": [],
        "raw_html_pages": [],
        "scraped_jobs": [],
        "extracted_jobs": [],
        "ranked_jobs": [],
        "final_report": "",
        "errors": [],
        "execution_log": [],
    }


async def graph_predict(inputs: dict) -> dict:
    """End-to-end run with injected FakeBrowser; records every tool call + node."""
    from graph.workflow import build_graph

    recorder: list[dict] = []
    try:
        graph = build_graph(
            checkpointer=InMemorySaver(),
            browser_tools=FakeBrowser(recorder),
        )
        thread_id = inputs.get("thread_id", "eval")
        config = {"configurable": {"thread_id": thread_id}}
        initial = _base_state(inputs)

        nodes_called: list[str] = []
        async for chunk in graph.astream(initial, config=config):
            for node_name in chunk:
                nodes_called.append(node_name)

        final = await graph.aget_state(config)
        values = final.values
        return {
            "status": "done",
            "final_report": values.get("final_report", ""),
            "num_jobs": len(values.get("ranked_jobs", [])),
            "tools_called": [c["tool"] for c in recorder],
            "tool_details": recorder,
            "nodes_called": nodes_called,
            "errors": values.get("errors", []),
        }
    except Exception as e:
        logger.error(f"[eval] graph_predict failed: {e}")
        return {
            "status": "error",
            "final_report": "",
            "num_jobs": 0,
            "tools_called": [c["tool"] for c in recorder],
            "tool_details": recorder,
            "nodes_called": [],
            "errors": [str(e)],
        }