"""Geração segura da apresentação inicial do atendimento."""

import re
import unicodedata
from collections.abc import Callable
from uuid import uuid4

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from banco_agil.domain.exceptions import IntegrationError
from banco_agil.integrations.llm import StructuredLlm
from banco_agil.prompts.registry import WELCOME_PROMPT_DEFINITION

DEFAULT_WELCOME_MESSAGE = (
    "Olá! Eu sou o assistente virtual do Banco Ágil. Posso consultar seu limite "
    "de crédito, solicitar aumento, conduzir uma entrevista de crédito e consultar "
    "cotações de moedas. Para começar, conte como posso ajudar."
)

_REQUIRED_TOPICS = ("banco agil", "limite", "aumento", "entrevista", "cotacao")
_FORBIDDEN_TOPICS = ("groq", "prompt", "tool", "agente interno")


class WelcomeReply(BaseModel):
    """Saída estruturada e limitada da apresentação inicial."""

    message: str = Field(min_length=40, max_length=400)


def generate_welcome_message(
    llm: StructuredLlm | None,
    turn_id_factory: Callable[[], str] = lambda: uuid4().hex,
) -> str:
    """Gera a apresentação sem contexto do cliente e usa fallback em falha."""
    if llm is None:
        return DEFAULT_WELCOME_MESSAGE

    try:
        result = llm.invoke_structured(
            f"welcome-{turn_id_factory()}",
            [SystemMessage(content=WELCOME_PROMPT_DEFINITION.template)],
            WelcomeReply,
            prompt_version=(
                f"{WELCOME_PROMPT_DEFINITION.prompt_id}"
                f"@{WELCOME_PROMPT_DEFINITION.version}"
            ),
        )
    except (IntegrationError, ValueError):
        return DEFAULT_WELCOME_MESSAGE

    message = " ".join(result.message.split())
    normalized = _normalized_text(message)
    if not all(topic in normalized for topic in _REQUIRED_TOPICS):
        return DEFAULT_WELCOME_MESSAGE
    if any(topic in normalized for topic in _FORBIDDEN_TOPICS):
        return DEFAULT_WELCOME_MESSAGE
    return message


def _normalized_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    without_accents = "".join(
        char for char in decomposed if not unicodedata.combining(char)
    )
    return re.sub(r"[^a-z0-9]+", " ", without_accents.casefold()).strip()
