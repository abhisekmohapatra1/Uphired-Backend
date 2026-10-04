"""Golden datasets for evals.

PROMPT_CASES  — snapshot each agent's real system prompt + representative input,
                with the expected output shape/golden for correctness checks.
GRAPH_CASES   — end-to-end inputs for the full LangGraph workflow (planner →
                browser+tools → extractor → ranker → summary) with expected
                tool-dispatch and output criteria.
"""

from agents.planner_agent import PLANNER_SYSTEM_PROMPT, BROADEN_SYSTEM_PROMPT
from agents.extractor_agent import ENRICHMENT_PROMPT
from agents.ranker_agent import RELEVANCE_PROMPT, SCORE_PROMPT
from agents.summary_agent import SUMMARY_PROMPT
from agents.resume_parser_agent import PARSER_PROMPT


# ── Prompt-level evaluation cases ────────────────────────────────────────────

PROMPT_CASES = [
    {
        "agent": "planner",
        "system_prompt": PLANNER_SYSTEM_PROMPT,
        "human_text": (
            "User request: senior python backend engineer\n"
            "Skills: python, fastapi, postgresql\n"
            "Experience: 5 years\n"
            "Market: global"
        ),
        "expected_shape": {
            "type": "object", "keys": ["queries"],
            "arrays": {"queries": {"min_items": 2, "max_items": 4,
                                   "item_keys": ["query", "reasoning"]}},
        },
        "golden": "2-4 distinct search queries each with a 'query' and 'reasoning', "
                  "covering the core role plus adjacent/skill-based variants.",
        "criterion": "Valid JSON object containing a 'queries' array of 2-4 items, each "
                     "with non-empty string 'query' and 'reasoning', all relevant to "
                     "a senior python backend engineer search.",
    },
    {
        "agent": "planner_broaden",
        "system_prompt": BROADEN_SYSTEM_PROMPT,
        "human_text": (
            "User request: senior python backend engineer\n"
            "Skills: python, fastapi, postgresql\n"
            "Experience: 5 years\n"
            "Market: global"
        ),
        "expected_shape": {
            "type": "object", "keys": ["queries"],
            "arrays": {"queries": {"min_items": 1, "max_items": 4,
                                   "item_keys": ["query", "reasoning"]}},
        },
        "golden": "2 simpler, broader 1-2 word queries (e.g. 'developer', 'engineer') "
                  "to cast a wider net after a failed search.",
        "criterion": "Valid JSON 'queries' array of simple, short (1-2 word) broad "
                     "search queries, each with a 'query' and 'reasoning'.",
    },
    {
        "agent": "extractor",
        "system_prompt": ENRICHMENT_PROMPT,
        "human_text": (
            "Title: Data Engineer\n"
            "Company: DataCo\n"
            "Description: Build ETL pipelines in Python and Spark, model in dbt, "
            "orchestrate with Airflow, deploy on AWS."
        ),
        "expected_shape": {"type": "list", "value_type": "str", "min_items": 1, "max_items": 8},
        "golden": "5-8 short, job-relevant skill strings (e.g. python, spark, airflow, aws, sql).",
        "criterion": "Output is a JSON array of 1-8 short lowercase skill strings "
                     "directly relevant to the Data Engineer role. No explanations, no markdown.",
    },
    {
        "agent": "ranker_relevance",
        "system_prompt": RELEVANCE_PROMPT,
        "human_text": (
            "User search intent: 'python backend developer'\n\n"
            "Job titles:\n"
            "0: Backend Engineer at Acme\n"
            "1: React Frontend Dev at Globex\n"
            "2: Registered Nurse at HealthCo\n"
            "3: Python Developer at Initech\n"
            "4: Accountant at Umbrella"
        ),
        "expected_shape": {"type": "list", "value_type": "int"},
        "golden": "Only the indices of genuinely tech-relevant jobs, e.g. [0, 3].",
        "criterion": "Output is a JSON array of 0-based indices of ONLY tech roles "
                     "relevant to a python backend search; non-tech or unrelated "
                     "roles must be excluded.",
    },
    {
        "agent": "ranker_score",
        "system_prompt": SCORE_PROMPT,
        "human_text": (
            "Candidate skills: python, fastapi, postgresql\n"
            "Candidate experience: 5 years\n"
            "Job title: Backend Engineer\n"
            "Job company: Acme\n"
            "Job skills: python, fastapi, kubernetes, aws\n"
            "Job description: Build backend services with python and fastapi, deploy on aws."
        ),
        "expected_shape": {
            "type": "object", "keys": ["score", "matched_skills", "missing_skills"],
            "score_range": [0.0, 1.0],
            "arrays": {"matched_skills": {"min_items": 0}, "missing_skills": {"min_items": 0}},
        },
        "golden": "score ~0.7-0.9 (strong overlap), matched_skills=[python, fastapi], "
                  "missing_skills=[kubernetes, aws].",
        "criterion": "Valid JSON object with numeric 'score' in [0,1] reflecting skill "
                     "overlap, and 'matched_skills'/'missing_skills' arrays consistent "
                     "with the given candidate and job skills.",
    },
    {
        "agent": "summary",
        "system_prompt": SUMMARY_PROMPT,
        "human_text": (
            "User query: python backend jobs\n"
            "User skills: python, fastapi\n"
            "Jobs found:\n"
            "- Backend Engineer at Acme | score: 0.9 | skills: python, fastapi | https://x.io/1\n"
            "- Python Developer at Globex | score: 0.8 | skills: python, docker | https://x.io/2\n"
            "Errors encountered: none"
        ),
        "expected_shape": {"type": "text", "min_char": 100},
        "golden": "Concise markdown report with job count, top matching jobs, matched and "
                  "missing skills, and one actionable recommendation.",
        "criterion": "Markdown report under 400 words with: total jobs, top jobs with "
                     "score/company/title/URL, matched + missing skills, and one "
                     "actionable recommendation.",
    },
    {
        "agent": "resume_parser",
        "system_prompt": PARSER_PROMPT,
        "human_text": (
            "Resume text:\n\n"
            "ARJUN MEHTA\narjun.mehta@gmail.com | +91 98765 43210 | Bengaluru, India\n"
            "Backend Engineer with 6 years building Python/FastAPI services, AWS, PostgreSQL.\n"
            "Experience: Senior Engineer, Paytm (Jan 2022 - Present); Engineer, Flipkart "
            "(Jun 2019 - Dec 2021).\nEducation: B.Tech Computer Science, IIT Bombay 2019."
        ),
        "expected_shape": {
            "type": "object", "keys": ["full_name", "email", "skills", "experience_years"],
            "numeric_min": {"experience_years": 5},
            "arrays": {"skills": {"min_items": 1}},
        },
        "golden": "full_name containing 'Mehta', email 'arjun.mehta@gmail.com', "
                  "experience_years >= 5, skills containing python and fastapi.",
        "criterion": "Valid JSON profile object with non-empty full_name/email, numeric "
                     "experience_years >= 5, and a skills array containing python and fastapi.",
    },
]


def prompt_examples() -> list[dict]:
    """Returns rows shaped for LangSmith: inputs + outputs."""
    rows = []
    for case in PROMPT_CASES:
        rows.append({
            "inputs": {
                "agent": case["agent"],
                "system_prompt": case["system_prompt"],
                "human_text": case["human_text"],
            },
            "outputs": {
                "expected_shape": case["expected_shape"],
                "golden": case["golden"],
                "criterion": case["criterion"],
            },
        })
    return rows


# ── End-to-end graph evaluation cases ────────────────────────────────────────

GRAPH_CASES = [
    {
        "user_query": "senior backend engineer python fastapi",
        "skills": ["python", "fastapi", "postgresql"],
        "experience_years": 5,
        "sites": ["remoteok", "linkedin"],
        "market": "global",
        "expected_sites": ["remoteok", "linkedin"],
        "min_jobs": 2,
        "golden_keywords": ["python", "fastapi"],
        "golden_report": (
            "A concise markdown report: total jobs found, top 5 matching jobs with "
            "match score/company/title/URL, matched skills, missing skills, and one "
            "actionable recommendation."
        ),
    },
    {
        "user_query": "devops engineer aws kubernetes",
        "skills": ["aws", "kubernetes", "terraform"],
        "experience_years": 3,
        "sites": ["indeed", "wellfound"],
        "market": "global",
        "expected_sites": ["indeed", "wellfound"],
        "min_jobs": 1,
        "golden_keywords": ["aws", "kubernetes"],
        "golden_report": (
            "A concise markdown report: total jobs found, top matching jobs with "
            "score/company/title/URL, matched skills, missing skills, and one "
            "actionable recommendation."
        ),
    },
]


def graph_examples() -> list[dict]:
    """Returns rows shaped for LangSmith: inputs + outputs."""
    rows = []
    for case in GRAPH_CASES:
        rows.append({
            "inputs": {
                "user_query": case["user_query"],
                "skills": case["skills"],
                "experience_years": case["experience_years"],
                "sites": case["sites"],
                "market": case["market"],
            },
            "outputs": {
                "expected_sites": case["expected_sites"],
                "min_jobs": case["min_jobs"],
                "golden_keywords": case["golden_keywords"],
                "golden_report": case["golden_report"],
            },
        })
    return rows