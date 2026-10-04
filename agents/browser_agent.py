import asyncio
from loguru import logger
from tools.browser_tools import BrowserTools
from graph.state import AgentState

SITE_HANDLERS = {
    "remoteok":         "search_remoteok",
    "linkedin":         "search_linkedin",
    "indeed":           "search_indeed",
    "naukri":           "search_naukri",
    "wellfound":        "search_wellfound",
    "ycombinator_jobs": "search_ycombinator",
    "ycombinator":      "search_ycombinator",
}


async def browser_node(state: AgentState, browser_tools=None) -> dict:
    logger.info("Browser agent starting")

    # Now stores list[dict] directly — real structured jobs with real URLs
    all_scraped_jobs = []
    errors = list(state.get("errors", []))
    log = list(state.get("execution_log", []))
    retry_count = state.get("retry_count", 0)

    # Injectable for evals (FakeBrowser); defaults to the real playwright tools
    tools = browser_tools if browser_tools is not None else BrowserTools()

    async with tools as browser:
        for task in state.get("search_plan", []):
            site = task.get("site", "").lower()
            query = task.get("query", "")

            handler_name = SITE_HANDLERS.get(site)
            if not handler_name:
                logger.warning(f"Unknown site '{site}', skipping")
                continue

            log_entry = f"[Browser] Searching '{query}' on {site}"
            logger.info(log_entry)
            log.append(log_entry)

            try:
                handler = getattr(browser, handler_name)
                jobs: list[dict] = await handler(query)

                if jobs:
                    all_scraped_jobs.extend(jobs)
                    log.append(f"[Browser] ✓ {site}: {len(jobs)} jobs ")
                else:
                    msg = f"[Browser] ✗ {site}: 0 jobs found"
                    logger.warning(msg)
                    errors.append(msg)

            except Exception as e:
                error_msg = f"[Browser] ✗ {site} crashed: {str(e)[:100]}"
                logger.error(error_msg)
                errors.append(error_msg)
                await asyncio.sleep(1)

    return {
        # Pass structured jobs directly — extractor just needs to enrich them
        "scraped_jobs": all_scraped_jobs,
        # Keep raw_html_pages empty — we no longer use HTML pipeline
        "raw_html_pages": ["__structured__"] if all_scraped_jobs else [],
        "retry_count": retry_count + (1 if not all_scraped_jobs else 0),
        "current_step": "extractor",
        "errors": errors,
        "execution_log": log,
    }