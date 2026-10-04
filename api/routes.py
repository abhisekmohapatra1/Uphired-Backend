import asyncio
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from graph.workflow import get_graph
from graph.state import AgentState
from loguru import logger

router = APIRouter()
active_workflows: dict[str, dict] = {}


class SearchRequest(BaseModel):
    query: str
    skills: List[str] = []
    experience_years: int = 0
    sites: Optional[List[str]] = None
    market: Optional[str] = "global"


class SearchResponse(BaseModel):
    workflow_id: str
    status: str


@router.post("/search-jobs", response_model=SearchResponse)
async def search_jobs(request: SearchRequest):
    workflow_id = str(uuid.uuid4())

    active_workflows[workflow_id] = {
        "status": "running",
        "logs": [],
        "result": None,
    }

    # asyncio.create_task keeps running even if the HTTP connection closes
    # Unlike BackgroundTasks which is tied to the request lifecycle
    asyncio.create_task(
        run_workflow(
            workflow_id=workflow_id,
            query=request.query,
            skills=request.skills,
            experience_years=request.experience_years,
            sites=request.sites,
            market=request.market,
        )
    )

    return SearchResponse(workflow_id=workflow_id, status="started")


async def run_workflow(
    workflow_id: str,
    query: str,
    skills: list,
    experience_years: int,
    sites: list = None,
    market: str = "global",
):
    graph = await get_graph()

    initial_state: AgentState = {
        "user_query": query,
        "user_skills": skills,
        "experience_years": experience_years,
        "preferred_sites": sites or [],
        "market": market,
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

    config = {"configurable": {"thread_id": workflow_id}}

    try:
        async for chunk in graph.astream(initial_state, config=config):
            for node_name, node_output in chunk.items():
                active_workflows[workflow_id]["logs"] = node_output.get(
                    "execution_log", active_workflows[workflow_id]["logs"]
                )
                active_workflows[workflow_id]["current_node"] = node_name

        final_state = await graph.aget_state(config)
        active_workflows[workflow_id]["status"] = "done"
        active_workflows[workflow_id]["result"] = {
            "report": final_state.values.get("final_report", ""),
            "jobs": [
                {
                    "title":    j.title,
                    "company":  j.company,
                    "score":    j.match_score,
                    "url":      j.url,
                    "skills":   j.skills,
                    "location": j.location,
                    "salary":   j.salary,
                    "source":   j.source,
                }
                for j in final_state.values.get("ranked_jobs", [])
            ],
        }
        logger.info(f"Workflow {workflow_id} completed — {len(active_workflows[workflow_id]['result']['jobs'])} jobs")

    except Exception as e:
        logger.error(f"Workflow {workflow_id} failed: {e}")
        active_workflows[workflow_id]["status"] = "error"
        active_workflows[workflow_id]["error"] = str(e)


@router.get("/job-results/{workflow_id}")
async def get_results(workflow_id: str):
    if workflow_id not in active_workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return active_workflows[workflow_id]