from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from dataclasses import dataclass, field


@dataclass
class JobListing:
    title: str = ""
    company: str = ""
    location: str = ""
    salary: str = ""
    skills: List[str] = field(default_factory=list)
    description: str = ""
    url: str = ""
    source: str = ""
    match_score: float = 0.0


class AgentState(TypedDict):
    # Input
    user_query: str
    user_skills: List[str]
    experience_years: int
    preferred_sites: List[str]       # ← NEW: from frontend site selector
    market: str                      # ← NEW: global/india/us/uk

    # Workflow control
    messages: Annotated[List[BaseMessage], add_messages]
    current_step: str
    retry_count: int
    max_retries: int
    search_broadened: bool

    # Agent outputs
    search_plan: List[dict]
    raw_html_pages: List[str]
    scraped_jobs: List[dict]
    extracted_jobs: List[JobListing]
    ranked_jobs: List[JobListing]
    final_report: str

    # Meta
    errors: List[str]
    execution_log: List[str]