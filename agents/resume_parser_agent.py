import json
import re
from pathlib import Path
from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage
from config import settings
from agents.llm_client import OpenRouterLLM

llm = OpenRouterLLM(
    model=settings.EXTRACTOR_MODEL,
    temperature=0,
)

PARSER_PROMPT = """
Extract ALL information from this resume text and return ONLY valid JSON.
No markdown, no explanation.

{
  "full_name": "",
  "email": "",
  "phone": "",
  "address": "",
  "city": "",
  "state": "",
  "country": "",
  "pincode": "",
  "linkedin_url": "",
  "github_url": "",
  "portfolio_url": "",
  "summary": "2-3 sentence professional summary",
  "experience_years": 0,
  "skills": ["Python", "FastAPI"],
  "work_history": [
    {
      "company": "",
      "role": "",
      "duration": "Jan 2022 - Present",
      "description": "What they did"
    }
  ],
  "education": [
    {
      "degree": "B.Tech Computer Science",
      "institution": "IIT Delhi",
      "year": "2022"
    }
  ],
  "certifications": ["AWS SAA"],
  "desired_roles": ["AI Engineer", "ML Engineer"],
  "notice_period": "Immediate"
}

Rules:
- If a field is not found, use empty string or empty array
- For experience_years: calculate from work history dates
- For desired_roles: infer from job titles and skills
- Extract every skill mentioned anywhere in the resume
"""


async def parse_resume(pdf_path: str) -> dict:
    """
    Extracts structured profile data from a resume PDF.
    Uses PyMuPDF to extract text, then LLM to parse into JSON.
    """
    try:
        import fitz   # PyMuPDF
    except ImportError:
        raise RuntimeError("Run: pip install pymupdf")

    # Extract text from all pages
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    if not full_text.strip():
        raise ValueError("Could not extract text from PDF. Is it a scanned image?")

    logger.info(f"Extracted {len(full_text)} chars from resume PDF")

    # Truncate if too long — LLMs have context limits
    resume_text = full_text[:8000]

    response = await llm.ainvoke([
        SystemMessage(content=PARSER_PROMPT),
        HumanMessage(content=f"Resume text:\n\n{resume_text}"),
    ])

    content = response.content.strip()
    # Strip markdown fences
    content = re.sub(r"```json|```", "", content).strip()

    try:
        profile = json.loads(content)
        logger.info(f"Resume parsed: {profile.get('full_name')} | {len(profile.get('skills', []))} skills")
        return profile
    except json.JSONDecodeError as e:
        logger.error(f"Resume parse JSON error: {e}")
        raise ValueError(f"LLM returned invalid JSON: {e}")