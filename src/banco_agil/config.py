"""Configuracao tipada carregada de ambiente local."""

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuracoes da aplicacao sem exigir credenciais em testes."""

    model_config = SettingsConfigDict(
        env_prefix="BANCO_AGIL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: SecretStr | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = Field(default=0.1, ge=0, le=1)
    llm_max_tokens: int = Field(default=180, ge=1)
    llm_timeout_seconds: float = Field(default=20.0, gt=0)
    awesomeapi_base_url: str = "https://economia.awesomeapi.com.br"
    data_dir: Path = Path("data")
    var_dir: Path = Path("var")
