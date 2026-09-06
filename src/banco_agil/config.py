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
    # Provedores aposentam modelos: o llama-3.3-70b saiu do catalogo da Groq e
    # toda chamada passou a falhar em silencio, caindo no texto canonico. Este
    # responde rapido, redige no tom do cliente e nao gasta tokens raciocinando.
    groq_model: str = "qwen/qwen3.8-27b"
    # Sobe de 0.3: a redacao final so muda a forma, e as guardas rejeitam o que
    # sair do texto validado, entao a variacao e barata e a resposta deixa de
    # sair sempre igual.
    llm_temperature: float = Field(default=0.5, ge=0, le=1)
    llm_max_tokens: int = Field(default=500, ge=1)
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    awesomeapi_base_url: str = "https://economia.awesomeapi.com.br"
    # Sem token a cota da AwesomeAPI e contada por IP, e em hospedagem
    # compartilhada esse IP e dividido com outras aplicacoes: a cota estoura por
    # uso alheio. Com token a contagem passa a ser da conta.
    awesomeapi_token: SecretStr | None = None
    data_dir: Path = Path("data")
    var_dir: Path = Path("var")
    conversation_checkpoint_path: Path | None = Field(
        default=None,
        description="SQLite opcional para checkpoints de sessão; None desabilita.",
    )
