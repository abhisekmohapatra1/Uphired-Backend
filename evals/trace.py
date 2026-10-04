import os
from loguru import logger
from config import settings


def init_tracing() -> bool:
    """
    Activates LangSmith tracing from env-configurable settings.
    Safe to call in prod and eval runners; no-op without an API key.
    """
    if not settings.LANGSMITH_API_KEY:
        logger.warning("LANGSMITH_API_KEY not set — tracing & evals disabled")
        return False
    os.environ.setdefault("LANGSMITH_API_KEY", settings.LANGSMITH_API_KEY)
    os.environ.setdefault("LANGSMITH_TRACING", settings.LANGSMITH_TRACING or "true")
    os.environ.setdefault("LANGSMITH_PROJECT", settings.LANGSMITH_PROJECT)
    logger.info(f"LangSmith tracing enabled → project '{settings.LANGSMITH_PROJECT}'")
    return True