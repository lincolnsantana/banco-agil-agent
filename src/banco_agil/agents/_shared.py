"""Comportamentos pequenos compartilhados pelos especialistas."""

import re
import unicodedata
from collections.abc import Sequence
from decimal import Decimal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from banco_agil.agents.state import ConversationState
from banco_agil.domain.enums import Agent, EndReason
from banco_agil.domain.exceptions import IntegrationError
from banco_agil.domain.models import EndServiceResult
from banco_agil.integrations.llm import StructuredLlm
from banco_agil.prompts.renderer import render_prompt
from banco_agil.tools.banking import end_service

_END_REQUESTS = {
    "cancelar atendimento",
    "encerrar",
    "encerrar atendimento",
    "fim",
    "parar",
    "quero encerrar",
    "nao quero continuar",
    "sair",
}
_END_PATTERN = re.compile(
    r"^(?:por favor,? )?(?:(?:eu )?(?:quero|gostaria de|pode) )?"
    r"(?:encerrar|encerre|finalizar|finalize|sair)(?: o atendimento)?$"
)
_SAFE_LLM_WORDS = frozenset(
    {
        "ajuda",
        "banco",
        "aumentar",
        "aumento",
        "cambio",
        "consultar",
        "cotacao",
        "dolar",
        "euro",
        "limite",
        "quero",
        "saber",
        "taxa",
        "cartao",
        "coisa",
        "credito",
        "emprestimo",
        "exterior",
        "financiamento",
        "moeda",
        "outra",
        "preciso",
        "resolver",
        "servico",
        "viagem",
    }
)


_UNSAFE_OPENING_WORDS = (
    "aprovad",
    "autenticad",
    "cotacao",
    "limite",
    "rejeitad",
    "score",
)


class HumanizedReply(BaseModel):
    """Abertura contextual sem fatos ou decisões bancárias."""

    opening: str = Field(min_length=1, max_length=100)


def normalized_text(value: str) -> str:
    """Normaliza texto curto para parsers deterministicos."""
    without_accents = "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold().strip())
        if not unicodedata.combining(character)
    )
    return " ".join(without_accents.split())


def end_reply_if_requested(state: ConversationState, user_text: str) -> str | None:
    """Encerra o atendimento antes de qualquer outra operacao."""
    normalized = normalized_text(user_text)
    if normalized not in _END_REQUESTS and _END_PATTERN.search(normalized) is None:
        return None
    end_conversation(state, EndReason.USER_REQUEST)
    return "Atendimento encerrado. Quando precisar, estaremos à disposição."


def end_conversation(state: ConversationState, reason: EndReason) -> None:
    """Executa a tool de encerramento e aplica seu resultado ao estado."""
    result = EndServiceResult.model_validate(end_service.invoke({"reason": reason}))
    state.end(result.reason)


def authentication_reply_if_missing(state: ConversationState) -> str | None:
    """Redireciona operacao protegida para coleta de credenciais."""
    if state.authenticated:
        return None
    state.active_agent = Agent.TRIAGE
    return "Para continuar, preciso confirmar seus dados. Por favor, informe seu CPF."


def parse_confirmation(value: str) -> bool | None:
    """Converte respostas curtas de consentimento sem usar LLM."""
    normalized = normalized_text(value)
    if normalized in {"sim", "aceito", "concordo", "pode", "quero"}:
        return True
    if normalized in {"nao", "recuso", "prefiro nao"}:
        return False
    return None


def sanitize_user_text(value: str) -> str:
    """Seleciona somente termos nao sensiveis antes de chamar o LLM."""
    words = re.findall(r"[a-z]+", normalized_text(value))
    return " ".join(word for word in words if word in _SAFE_LLM_WORDS)


def humanize_reply(
    state: ConversationState,
    canonical_reply: str,
    llm: StructuredLlm | None,
    turn_id: str,
    *,
    responding_agent: Agent | None,
    recent_messages: Sequence[BaseMessage],
    user_text: str,
) -> str:
    """Acrescenta uma abertura do LLM sem permitir mudança nos fatos.

    A resposta canônica permanece integral. Falhas, saídas inseguras e orçamento
    já consumido retornam silenciosamente ao texto determinístico.
    """
    if (
        llm is None
        or not turn_id
        or state.ended
        or responding_agent is None
        or llm.was_called(turn_id)
    ):
        return canonical_reply

    prompt_state = state.model_copy(deep=True)
    prompt_state.active_agent = responding_agent
    rendered = render_prompt(prompt_state)
    safe_user_text = sanitize_user_text(user_text) or "pedido bancario"
    messages: list[BaseMessage] = [
        rendered.system_message,
        *_safe_history(recent_messages),
        HumanMessage(
            content=(
                "Gere somente uma abertura acolhedora e contextual para a resposta "
                "já validada pelo sistema. Não inclua fatos, números, decisões, "
                "promessas ou perguntas. Contexto seguro: "
                f"{safe_user_text}."
            )
        ),
    ]
    try:
        result = llm.invoke_structured(
            turn_id,
            messages,
            HumanizedReply,
            prompt_version=rendered.prompt_version,
        )
    except IntegrationError:
        return canonical_reply

    opening = " ".join(result.opening.split())
    normalized_opening = normalized_text(opening)
    if (
        not opening
        or any(character.isdigit() for character in opening)
        or "?" in opening
        or any(
            unsafe_word in normalized_opening for unsafe_word in _UNSAFE_OPENING_WORDS
        )
    ):
        return canonical_reply
    return f"{opening} {canonical_reply}"


def _safe_history(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    safe_messages: list[BaseMessage] = []
    for message in messages[-5:]:
        safe_content = sanitize_user_text(str(message.content))
        if not safe_content:
            continue
        message_type = AIMessage if isinstance(message, AIMessage) else HumanMessage
        safe_messages.append(message_type(content=safe_content))
    return safe_messages


def format_money(value: Decimal) -> str:
    """Formata valor decimal para exibicao bancaria em pt-BR."""
    formatted = f"{value:,.2f}"
    return formatted.replace(",", "_").replace(".", ",").replace("_", ".")
