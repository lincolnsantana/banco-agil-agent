"""Comportamentos pequenos compartilhados pelos especialistas."""

import re
import unicodedata
from decimal import Decimal

from banco_agil.agents.state import ConversationState
from banco_agil.domain.enums import Agent, EndReason
from banco_agil.domain.models import EndServiceResult
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


def format_money(value: Decimal) -> str:
    """Formata valor decimal para exibicao bancaria em pt-BR."""
    formatted = f"{value:,.2f}"
    return formatted.replace(",", "_").replace(".", ",").replace("_", ".")
