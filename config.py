from pathlib import Path
from pydantic_settings import BaseSettings

ENV_FILE = Path(__file__).parent / ".env"


class Settings(BaseSettings):
    # Required
    OPENROUTER_API_KEY: str

    # Optional fallback keys — comma-separated list used for failover on 429/no-credits
    OPENROUTER_API_KEYS: str = ""

    # LLM settings
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    # Judge model used for LLM-as-a-judge evals (OpenRouter-compatible id)
    EVAL_MODEL: str = "openai/gpt-4o-mini"

    # LangSmith observability / eval tracing
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "uphired-backend"
    LANGSMITH_TRACING: str = "false"
    PLANNER_MODEL: str = "deepseek/deepseek-chat"
    EXTRACTOR_MODEL: str = "deepseek/deepseek-chat"
    SUMMARY_MODEL: str = "deepseek/deepseek-chat"

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./job_hunt.db"

    # App
    LOG_LEVEL: str = "DEBUG"
    APP_ENV: str = "development"

    # ← ADD THIS LINE
    TARGET_MARKET: str = "global"   # change to "india" to include Naukri searches

    model_config = {
        "env_file": str(ENV_FILE),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()