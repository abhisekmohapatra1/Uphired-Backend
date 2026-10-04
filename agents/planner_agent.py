import json
from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import AgentState
from config import settings
from agents.llm_client import OpenRouterLLM

llm = OpenRouterLLM(
    model=settings.PLANNER_MODEL,
    temperature=0.1,
)

PLANNER_SYSTEM_PROMPT = """
You are a job search query optimizer for the tech industry.

Given a user's job search request, generate MULTIPLE search queries to maximize coverage.

Rules:
- Generate 2-4 queries, each 2-4 words
- Use standard industry job titles
- Cover related roles (e.g. "backend developer" + "python engineer" + "software engineer")
- Include location-aware variants if market is specified
- Use synonyms to catch different job posting styles

Return ONLY valid JSON, no markdown:
{
  "queries": [
    {"query": "software engineer", "reasoning": "Core role, broadest match"},
    {"query": "python developer", "reasoning": "Skills-based variant"},
    {"query": "backend engineer", "reasoning": "Architecture variant"},
    {"query": "full stack developer", "reasoning": "Adjacent role, high overlap"}
  ]
}
"""

BROADEN_SYSTEM_PROMPT = """
You are a job search broadener. The previous search returned no or few results.

Generate 2 simpler, broader queries (1-2 words each) cast a wider net.

Return ONLY valid JSON, no markdown:
{
  "queries": [
    {"query": "developer", "reasoning": "Broadest match"},
    {"query": "engineer", "reasoning": "Catches related roles"}
  ]
}
"""


async def planner_node(state: AgentState) -> dict:
    logger.info("Planner agent starting")

    query = state["user_query"]
    broadened = state.get("search_broadened", False)

    all_skills = state.get("user_skills", [])
    experience = state.get("experience_years", 0)
    market = state.get("market", "global")

    user_message = f"""
User request: {query}
Skills: {', '.join(all_skills)}
Experience: {experience} years
Market: {market}
"""

    system_prompt = BROADEN_SYSTEM_PROMPT if broadened else PLANNER_SYSTEM_PROMPT

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ])

    try:
        content = response.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        plan = json.loads(content)
        queries = plan.get("queries", [{"query": query, "reasoning": "Original query"}])
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Planner JSON parse failed: {e}")
        words = query.split()[:3]
        queries = [{"query": " ".join(words), "reasoning": "Fallback plan"}]

    # Use EXACTLY the sites the user selected
    preferred_sites = state.get("preferred_sites", [])
    if not preferred_sites:
        preferred_sites = ["linkedin", "remoteok", "indeed"]

    # Create a search plan: each query × each site (but rotate queries across sites for diversity)
    search_plan = []
    for i, site in enumerate(preferred_sites):
        query_entry = queries[i % len(queries)]
        search_plan.append({
            "site": site,
            "query": query_entry["query"],
            "filters": {},
        })

    query_summary = " | ".join([q["query"] for q in queries])
    log_entry = f"[Planner] Queries: {query_summary} → sites: {preferred_sites} | skills: {len(all_skills)} loaded"
    logger.info(log_entry)
    for q in queries:
        logger.info(f"[Planner]   - '{q['query']}': {q.get('reasoning', '')}")

    return {
        "search_plan": search_plan,
        "current_step": "browser",
        "retry_count": 0,
        "execution_log": state.get("execution_log", []) + [log_entry],
        "messages": [response],
    }
