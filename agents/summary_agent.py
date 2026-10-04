from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import AgentState
from config import settings
from agents.llm_client import OpenRouterLLM


llm = OpenRouterLLM(
    model=settings.SUMMARY_MODEL,
    temperature=0.3,
)

SUMMARY_PROMPT = """
You are a career assistant. Generate a concise markdown job search report.
Include:
- Total jobs found and search summary
- Top 5 jobs with match score, company, title, URL
- Skills you matched
- Skills you're missing (based on jobs that didn't match well)
- One actionable recommendation

Keep it under 400 words. Use markdown formatting.
"""


async def summary_node(state: AgentState) -> dict:
    """
    Summary agent — generates a human-readable markdown report.
    """
    logger.info("Summary agent starting")

    ranked_jobs = state.get("ranked_jobs", [])
    
    if not ranked_jobs:
        return {
            "final_report": "## No jobs found\n\nTry broadening your search or adjusting your skills.",
            "current_step": "done",
        }

    # Build compact job list for LLM (avoid token bloat)
    jobs_summary = "\n".join([
        f"- {j.title} at {j.company} | score: {j.match_score} | skills: {', '.join(j.skills[:5])} | {j.url}"
        for j in ranked_jobs[:10]
    ])

    user_message = f"""
    User query: {state['user_query']}
    User skills: {', '.join(state.get('user_skills', []))}
    
    Jobs found:
    {jobs_summary}
    
    Errors encountered: {', '.join(state.get('errors', [])) or 'none'}
    """

    response = await llm.ainvoke([
        SystemMessage(content=SUMMARY_PROMPT),
        HumanMessage(content=user_message),
    ])

    log_entry = "[Summary] Report generated"
    logger.info(log_entry)

    return {
        "final_report": response.content,
        "current_step": "done",
        "execution_log": state.get("execution_log", []) + [log_entry],
    }