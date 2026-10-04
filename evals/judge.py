"""LLM-as-a-judge evaluation helper.

Uses the existing OpenRouter failover client so judge calls inherit
API-key rotation (429 / no-credits retries) automatically.
"""

import json
import re
from loguru import logger
from langchain_core.messages import SystemMessage, HumanMessage
from agents.llm_client import OpenRouterLLM
from config import settings

_judge = OpenRouterLLM(model=settings.EVAL_MODEL, temperature=0)

JUDGE_SYSTEM_PROMPT = """
You are a meticulous LLM evaluation judge. Grade the OUTPUT against the CRITERION,
optionally cross-checking the REFERENCE. Be strict, calibrated, and consistent.

Return ONLY valid JSON, no markdown:
{"score": <int 0-10>, "reason": "<1-2 sentence justification>"}

Rubric:
- 10: perfect — fully meets the criterion, completely faithful to the reference
- 7-9: excellent — only minor omissions
- 5-6: acceptable — noticeable gaps or minor inaccuracies
- 3-4: partially wrong, missing key points expected by the reference
- 1-2: largely incorrect or off-topic
- 0: empty, unrelated, or directly contradicts the reference
"""


def _strip_fences(text: str) -> str:
    return re.sub(r"```(?:json)?", "", text).strip()


async def judge(criterion: str, output: str, reference: str = "") -> dict:
    """Returns {"score": 0.0-1.0, "reason": str}. Fails safe to 0.0 with a reason."""
    ref = reference.strip() or "No explicit reference — judge absolute quality against the criterion."
    messages = [
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"CRITERION:\n{criterion}\n\n"
            f"REFERENCE (expected):\n{ref}\n\n"
            f"OUTPUT (to grade):\n{output}"
        )),
    ]
    try:
        response = await _judge.ainvoke(messages)
        data = json.loads(_strip_fences(response.content))
        score = max(0, min(int(data.get("score", 0)), 10))
        return {"score": score / 10.0, "reason": str(data.get("reason", ""))}
    except Exception as e:
        logger.warning(f"Judge call failed: {e}")
        return {"score": 0.0, "reason": f"judge error: {e}"}