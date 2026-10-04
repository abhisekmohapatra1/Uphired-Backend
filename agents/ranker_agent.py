import json
from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import AgentState, JobListing
from config import settings
from agents.llm_client import OpenRouterLLM

llm = OpenRouterLLM(
    model=settings.PLANNER_MODEL,
    temperature=0,
)

RELEVANCE_PROMPT = """
You are a job relevance filter for TECH jobs. Given a user's job search intent and a list of job titles,
return ONLY the indices of jobs that are genuinely relevant to tech roles.

A job is relevant if:
- It's a tech/engineering/IT role (software, data, devops, design, product, QA, security, etc.)
- The title is in the same field or a closely related tech field
- It could reasonably appear in search results for that query

A job is NOT relevant if:
- It is in a completely unrelated field (e.g. nursing, accounting, sales non-tech)
- It shares only a company name or location, not the actual role

Return ONLY a JSON array of relevant indices (0-based). No explanation, no markdown.
Example: [0, 2, 3, 7]
"""

SCORE_PROMPT = """
You are a senior tech recruiter. Score how well a candidate's profile matches a job posting.

Consider:
- Skills overlap (most important)
- Experience level alignment
- Role type match (frontend vs backend vs fullstack vs data vs devops)
- Title relevance

Score rules:
- 0.9-1.0: near-perfect match, candidate has almost all required skills
- 0.7-0.89: strong match, candidate has most key skills
- 0.5-0.69: decent match, candidate has some skills, missing a few
- 0.3-0.49: partial match, foundational skills but missing specifics
- 0.1-0.29: weak match, very few relevant skills
- 0.0: no match

Return ONLY a JSON object, no markdown:
{"score": 0.75, "matched_skills": ["python", "fastapi"], "missing_skills": ["kubernetes", "aws"]}
"""


# Skill similarity groups — skills that are semantically similar
SKILL_GROUPS = {
    "python": ["python3", "python2", "cpython", "pypy", "django", "flask", "fastapi", "uvicorn", "gunicorn", "celery"],
    "javascript": ["js", "typescript", "ts", "es6", "es2015", "ecmascript", "node", "nodejs", "deno", "bun"],
    "react": ["reactjs", "react.js", "next", "nextjs", "next.js", "remix", "gatsby"],
    "vue": ["vuejs", "vue.js", "nuxt", "nuxtjs"],
    "angular": ["angularjs", "angular.js", "ng"],
    "java": ["jvm", "jdk", "spring", "springboot", "spring-boot", "hibernate", "maven", "gradle"],
    "golang": ["go", "golang"],
    "rust": ["rustlang", "cargo"],
    "sql": ["mysql", "postgresql", "postgres", "sqlite", "mssql", "oracle", "database"],
    "nosql": ["mongodb", "redis", "cassandra", "dynamodb", "couchdb", "elasticsearch"],
    "aws": ["amazon web services", "s3", "ec2", "lambda", "sqs", "sns", "cloudformation", "ecs", "eks"],
    "gcp": ["google cloud", "bigquery", "cloud functions", "cloud run", "gke"],
    "azure": ["microsoft azure", "azure devops", "azure functions"],
    "docker": ["containerization", "containers", "podman"],
    "kubernetes": ["k8s", "helm", "istio", "openshift"],
    "git": ["github", "gitlab", "bitbucket", "version control"],
    "linux": ["unix", "bash", "shell", "posix"],
    "ml": ["machine learning", "deep learning", "pytorch", "tensorflow", "scikit-learn", "sklearn", "keras", "huggingface", "transformers"],
    "data science": ["data analysis", "pandas", "numpy", "jupyter", "notebook", "matplotlib", "seaborn"],
    "devops": ["cicd", "ci/cd", "terraform", "ansible", "jenkins", "github actions", "gitlab ci"],
    "frontend": ["ui", "ux", "css", "scss", "sass", "tailwind", "html", "web development"],
    "backend": ["server side", "api", "rest", "grpc", "microservices", "distributed systems"],
    "mobile": ["android", "ios", "react native", "flutter", "swift", "kotlin"],
}


def _expand_skill(skill: str) -> set:
    """Expand a skill into its similarity group."""
    skill_lower = skill.lower().strip()
    expanded = {skill_lower}
    for group_key, group_members in SKILL_GROUPS.items():
        if skill_lower == group_key or skill_lower in group_members:
            expanded.add(group_key)
            expanded.update(m for m in group_members)
    return expanded


def _semantic_skill_score(job_skills: list[str], user_skills: list[str]) -> float:
    """Score based on expanded skill similarity groups."""
    if not job_skills or not user_skills:
        return 0.1

    job_expanded = set()
    for s in job_skills:
        job_expanded.update(_expand_skill(s))

    user_expanded = set()
    for s in user_skills:
        user_expanded.update(_expand_skill(s))

    if not job_expanded:
        return 0.1

    matched = job_expanded & user_expanded
    # Weight: matched / total job requirements
    score = len(matched) / max(len(job_expanded), 1)
    return round(min(score, 1.0), 2)


def _title_similarity(title: str, user_query: str) -> float:
    """Boost score if job title matches the user's search intent."""
    title_lower = title.lower()
    query_lower = user_query.lower()
    query_words = set(query_lower.split())

    # Direct word overlap
    title_words = set(title_lower.split())
    overlap = query_words & title_words
    if overlap:
        return min(len(overlap) / max(len(query_words), 1), 1.0)

    # Check if query words appear as substrings in title
    for word in query_words:
        if len(word) > 3 and word in title_lower:
            return 0.5
    return 0.0


async def _filter_relevant_jobs(
    jobs: list[JobListing],
    user_query: str,
) -> list[JobListing]:
    """
    Ask LLM which jobs are actually relevant to the search intent.
    Processes in batches of 20 to stay within token limits.
    """
    if not jobs:
        return []

    relevant = []
    batch_size = 20

    for i in range(0, len(jobs), batch_size):
        batch = jobs[i:i + batch_size]
        titles_list = "\n".join([
            f"{j}: {batch[j].title} at {batch[j].company}"
            for j in range(len(batch))
        ])

        try:
            response = await llm.ainvoke([
                SystemMessage(content=RELEVANCE_PROMPT),
                HumanMessage(content=f"User search intent: '{user_query}'\n\nJob titles:\n{titles_list}"),
            ])

            content = response.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            relevant_indices = json.loads(content)

            for idx in relevant_indices:
                if 0 <= idx < len(batch):
                    relevant.append(batch[idx])

        except Exception as e:
            logger.warning(f"Relevance filter failed for batch: {e} — keeping all jobs in batch")
            relevant.extend(batch)

    logger.info(f"Relevance filter: {len(jobs)} → {len(relevant)} jobs")
    return relevant


async def ranker_node(state: AgentState) -> dict:
    logger.info("Ranker agent starting")

    user_skills = state.get("user_skills", [])
    user_query = state.get("user_query", "")
    jobs = state.get("extracted_jobs", [])

    if not jobs:
        return {
            "ranked_jobs": [],
            "current_step": "summary",
            "execution_log": state.get("execution_log", []) + ["[Ranker] No jobs to rank"],
        }

    # Step 1: LLM filters out irrelevant jobs
    relevant_jobs = await _filter_relevant_jobs(jobs, user_query)

    # Step 2: Multi-factor scoring
    for job in relevant_jobs:
        # Factor 1: Semantic skill match (60% weight)
        skill_score = _semantic_skill_score(job.skills, user_skills)

        # Factor 2: Title relevance (25% weight)
        title_score = _title_similarity(job.title, user_query)

        # Factor 3: Description keyword bonus (15% weight)
        desc_bonus = 0.0
        if job.description:
            desc_lower = job.description.lower()
            for skill in user_skills[:10]:
                if skill.lower() in desc_lower:
                    desc_bonus += 0.02
            desc_bonus = min(desc_bonus, 0.15)

        # Combined score
        job.match_score = round(
            skill_score * 0.60 + title_score * 0.25 + desc_bonus, 2
        )

    # Step 3: Sort by score descending, deduplicate by title+company
    seen = set()
    deduped = []
    for job in sorted(relevant_jobs, key=lambda j: j.match_score, reverse=True):
        key = (job.title.lower().strip(), job.company.lower().strip())
        if key not in seen:
            seen.add(key)
            deduped.append(job)

    # Step 4: LLM-based scoring for top 15 jobs (most accurate, but costly)
    top_jobs = deduped[:15]
    if top_jobs and user_skills:
        try:
            for job in top_jobs:
                if job.match_score >= 0.3:  # Only score jobs with some match
                    response = await llm.ainvoke([
                        SystemMessage(content=SCORE_PROMPT),
                        HumanMessage(content=f"""
Candidate skills: {', '.join(user_skills)}
Candidate experience: {state.get('experience_years', 0)} years
Job title: {job.title}
Job company: {job.company}
Job skills: {', '.join(job.skills[:10])}
Job description: {job.description[:500] if job.description else 'N/A'}
"""),
                    ])
                    content = response.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
                    result = json.loads(content)
                    # Blend LLM score with keyword score for robustness
                    llm_score = result.get("score", job.match_score)
                    job.match_score = round((job.match_score + llm_score) / 2, 2)
                    if result.get("matched_skills"):
                        job.skills = list(set(job.skills + result["matched_skills"]))
        except Exception as e:
            logger.warning(f"LLM scoring failed: {e} — using keyword scores")

    ranked = deduped

    log_entry = (
        f"[Ranker] {len(jobs)} total → {len(relevant_jobs)} relevant → "
        f"{len(ranked)} unique → top score: {ranked[0].match_score if ranked else 0}"
    )
    logger.info(log_entry)

    return {
        "ranked_jobs": ranked,
        "current_step": "summary",
        "execution_log": state.get("execution_log", []) + [log_entry],
    }
