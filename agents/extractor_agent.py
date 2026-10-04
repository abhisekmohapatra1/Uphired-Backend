import json
from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import AgentState, JobListing
from config import settings
from agents.llm_client import OpenRouterLLM

llm = OpenRouterLLM(
    model=settings.EXTRACTOR_MODEL,
    temperature=0,
)

ENRICHMENT_PROMPT = """
Given this job listing title, company, and description, infer the key skills,
tools, or qualifications this role likely requires.

This could be ANY profession — tech, healthcare, finance, design, marketing, etc.
Return skills/qualifications relevant to THAT specific role.

Return ONLY a JSON array of short skill strings, no explanation, no markdown.
Examples:
- Software role: ["python", "aws", "docker", "sql"]
- Marketing role: ["seo", "google analytics", "copywriting", "campaign management"]  
- Nurse role: ["patient care", "icu", "bls certification", "emr systems"]
- Finance role: ["excel", "financial modeling", "cfa", "risk analysis"]

5-8 skills maximum.
"""


async def _enrich_skills(job: dict) -> list[str]:
    """Infer skills from job title + description. Works for any profession."""
    if job.get("skills"):
        return job["skills"]

    try:
        response = await llm.ainvoke([
            SystemMessage(content=ENRICHMENT_PROMPT),
            HumanMessage(content=(
                f"Title: {job.get('title', '')}\n"
                f"Company: {job.get('company', '')}\n"
                f"Description: {job.get('description', '')}"
            )),
        ])
        content = response.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        skills = json.loads(content)
        return [s.lower().strip() for s in skills if isinstance(s, str)]
    except Exception as e:
        logger.debug(f"Skill enrichment failed: {e}")
        # Fallback: use meaningful words from title
        stopwords = {"and", "or", "the", "for", "with", "at", "in", "of", "to", "a"}
        return [
            w.lower() for w in job.get("title", "").split()
            if len(w) > 3 and w.lower() not in stopwords
        ]


async def extractor_node(state: AgentState) -> dict:
    logger.info("Extractor agent starting")

    log = list(state.get("execution_log", []))
    errors = list(state.get("errors", []))
    scraped_jobs = state.get("scraped_jobs", [])

    if not scraped_jobs:
        log.append("[Extractor] No scraped jobs to process")
        return {
            "extracted_jobs": [],
            "search_broadened": True,
            "current_step": "ranker",
            "errors": errors,
            "execution_log": log,
        }

    all_jobs: list[JobListing] = []

    for job_dict in scraped_jobs:
        url = job_dict.get("url", "")
        if not url.startswith("http"):
            continue

        skills = await _enrich_skills(job_dict)

        job = JobListing(
            title=job_dict.get("title", "").strip(),
            company=job_dict.get("company", "").strip(),
            location=job_dict.get("location", "").strip(),
            salary=job_dict.get("salary", "").strip(),
            skills=skills,
            description=job_dict.get("description", "")[:300],
            url=url,
            source=job_dict.get("source", ""),
        )

        if job.title:
            all_jobs.append(job)

    log_entry = f"[Extractor] Processed {len(all_jobs)} jobs"
    logger.info(log_entry)
    log.append(log_entry)

    return {
        "extracted_jobs": all_jobs,
        "search_broadened": len(all_jobs) == 0,
        "current_step": "ranker",
        "errors": errors,
        "execution_log": log,
    }