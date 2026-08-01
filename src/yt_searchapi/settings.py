"""Application settings loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict

# These apply to new projects unless the user explicitly chooses worker counts.
# Persisted project checkpoints retain the values they were created with.
DEFAULT_SEARCHAPI_WORKERS = 8
DEFAULT_LLM_WORKERS = 16


class Settings(BaseSettings):
    """Credentials loaded from environment variables or an untracked .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    searchapi_api_key: str | None = None
    openai_api_key: str | None = None
