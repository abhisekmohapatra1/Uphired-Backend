import threading
import time
from loguru import logger
from langchain_openai import ChatOpenAI
from config import settings

RETRYABLE_HTTP_CODES = (429, 500, 502, 503, 504)

RETRYABLE_MSG_MARKERS = (
    "rate_limit",
    "rate limit",
    "insufficient credits",
    "insufficient_credits",
    "no credits",
    "not enough credits",
    "out of credits",
    "quota",
    "credit limit",
    "could not verify available credits",
    "retry shortly",
)

# Keys that just got rate-limited are temporarily inactive to avoid
# hammering them on the very next call.
COOLDOWN_SECONDS = 60


def _load_api_keys() -> list[str]:
    raw = getattr(settings, "OPENROUTER_API_KEYS", None) or settings.OPENROUTER_API_KEY
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    return keys or [settings.OPENROUTER_API_KEY]


_API_KEYS = _load_api_keys()
_next_key_index = 0
_index_lock = threading.Lock()
_degraded_until: dict[str, float] = {}


def _pick_key() -> str:
    global _next_key_index
    now = time.monotonic()
    n = len(_API_KEYS)
    with _index_lock:
        for offset in range(n):
            idx = (_next_key_index + offset) % n
            key = _API_KEYS[idx]
            if _degraded_until.get(key, 0) <= now:
                _next_key_index = (idx + 1) % n
                return key
        key = _API_KEYS[_next_key_index]
        _next_key_index = (_next_key_index + 1) % n
        return key


def _mark_degraded(key: str) -> None:
    _degraded_until[key] = time.monotonic() + COOLDOWN_SECONDS


def _is_retryable_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is not None:
        try:
            if int(status) in RETRYABLE_HTTP_CODES:
                return True
        except (TypeError, ValueError):
            pass

    body = getattr(exc, "body", None)
    if body is None and hasattr(exc, "response") and hasattr(exc.response, "json"):
        try:
            body = exc.response.json()
        except Exception:
            body = None

    if body:
        if isinstance(body, dict):
            error = body.get("error") if isinstance(body.get("error"), dict) else {}
            code = body.get("code") or error.get("code")
            if code is not None:
                try:
                    if int(code) == 429:
                        return True
                except (TypeError, ValueError):
                    pass
        text = str(body).lower()
    else:
        text = str(exc).lower()

    return any(marker in text for marker in RETRYABLE_MSG_MARKERS)


def _build_llm(model: str, temperature: float, api_key: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        base_url=settings.OPENROUTER_BASE_URL,
        api_key=api_key,
        temperature=temperature,
    )


class OpenRouterLLM:
    """Drop-in replacement for ChatOpenAI with API-key failover on retryable errors."""

    def __init__(self, model: str, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    async def ainvoke(self, messages, **kwargs):
        last_error = None
        for attempt in range(len(_API_KEYS)):
            api_key = _pick_key()
            llm = _build_llm(self.model, self.temperature, api_key)
            try:
                return await llm.ainvoke(messages, **kwargs)
            except Exception as exc:
                last_error = exc
                if not _is_retryable_error(exc):
                    raise
                _mark_degraded(api_key)
                logger.warning(
                    f"OpenRouter call failed with key #{attempt + 1}: {exc} — "
                    f"retrying with another key"
                )
        logger.error(
            f"All {len(_API_KEYS)} OpenRouter API keys failed: {last_error}"
        )
        raise last_error