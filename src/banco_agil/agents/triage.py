"""Nó de triagem integralmente determinístico."""

import re
from datetime import date

from banco_agil.agents._shared import (
    end_conversation,
    end_reply_if_requested,
    normalized_text,
)
from banco_agil.agents.state import ConversationState
from banco_agil.domain.enums import Agent, EndReason, Intent
from banco_agil.domain.exceptions import RepositoryError
from banco_agil.domain.models import AuthenticationResult
from banco_agil.services.authentication import AuthenticationService
from banco_agil.tools.banking import authenticate_client


def handle_triage(
    state: ConversationState,
    user_text: str,
    service: AuthenticationService,
) -> str:
    """Processa um turno de triagem e atualiza o estado confiavel."""
    end_reply = end_reply_if_requested(state, user_text)
    if end_reply is not None:
        return end_reply

    if not state.authenticated:
        return _handle_authentication(state, user_text, service)

    intent = _deterministic_intent(user_text)

    if intent is None or intent in {Intent.UNKNOWN, Intent.OTHER}:
        return "Posso ajudar com limite de crédito ou cotação de moedas. O que deseja?"
    if intent is Intent.END_SERVICE:
        end_conversation(state, EndReason.USER_REQUEST)
        return "Atendimento encerrado. Quando precisar, estaremos à disposição."

    state.intent = intent
    state.active_agent = (
        Agent.EXCHANGE if intent is Intent.EXCHANGE_RATE else Agent.CREDIT
    )
    return "Certo. Vou prosseguir com sua solicitação."


def _handle_authentication(
    state: ConversationState,
    user_text: str,
    service: AuthenticationService,
) -> str:
    if state.pending_cpf is None:
        cpf = re.sub(r"[.\-\s]", "", user_text)
        if not re.fullmatch(r"\d{11}", cpf):
            return "Olá! Para começar, informe seu CPF com 11 dígitos."
        state.pending_cpf = cpf
        return "Agora informe sua data de nascimento no formato AAAA-MM-DD."

    try:
        birth_date = date.fromisoformat(user_text.strip())
        if birth_date > date.today():
            raise ValueError
        state.pending_birth_date = birth_date
    except ValueError:
        state.pending_birth_date = None

    try:
        result = AuthenticationResult.model_validate(
            authenticate_client.invoke(
                {
                    "cpf": state.pending_cpf,
                    "birth_date": user_text.strip(),
                    "state": state,
                    "service": service,
                }
            )
        )
    except RepositoryError:
        return "Não foi possível validar os dados agora. Tente novamente mais tarde."
    if result.authenticated:
        return "Dados confirmados. Como posso ajudar hoje?"

    state.pending_cpf = None
    state.pending_birth_date = None
    if result.should_end:
        end_conversation(state, EndReason.AUTHENTICATION_FAILURES)
        return (
            "Não foi possível validar os dados após três tentativas. "
            "Atendimento encerrado."
        )
    return "Não foi possível validar os dados. Tente novamente informando seu CPF."


def _deterministic_intent(user_text: str) -> Intent | None:
    normalized = normalized_text(user_text)
    if any(word in normalized for word in ("cambio", "cotacao", "dolar", "euro")):
        return Intent.EXCHANGE_RATE
    if any(word in normalized for word in ("aumentar", "aumento", "novo limite")):
        return Intent.LIMIT_INCREASE
    if "limite" in normalized:
        return Intent.CREDIT_LIMIT
    return None
