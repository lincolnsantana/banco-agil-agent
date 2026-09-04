"""No de triagem com LLM primeiro e fallback deterministico."""

import re
from collections.abc import Sequence
from datetime import date

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel

from banco_agil.agents._shared import (
    end_conversation,
    end_reply_if_requested,
    normalized_text,
    sanitize_user_text,
)
from banco_agil.agents.state import ConversationState
from banco_agil.domain.enums import Agent, EndReason, Intent
from banco_agil.domain.exceptions import IntegrationError, RepositoryError
from banco_agil.domain.models import AuthenticationResult
from banco_agil.integrations.llm import StructuredLlm
from banco_agil.prompts.renderer import render_prompt
from banco_agil.services.authentication import AuthenticationService
from banco_agil.tools.banking import authenticate_client


class IntentDecision(BaseModel):
    """Saida estruturada para intencao livre ainda inconclusiva."""

    intent: Intent


def handle_triage(
    state: ConversationState,
    user_text: str,
    service: AuthenticationService,
    *,
    llm: StructuredLlm | None = None,
    turn_id: str = "",
    recent_messages: Sequence[BaseMessage] = (),
) -> str:
    """Processa um turno de triagem e atualiza o estado confiavel."""
    end_reply = end_reply_if_requested(state, user_text)
    if end_reply is not None:
        return end_reply

    if not state.authenticated:
        return _handle_authentication(state, user_text, service)

    intent = _llm_intent(state, user_text, llm, turn_id, recent_messages)
    if intent is None:
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


def _llm_intent(
    state: ConversationState,
    user_text: str,
    llm: StructuredLlm | None,
    turn_id: str,
    recent_messages: Sequence[BaseMessage],
) -> Intent | None:
    """Classifica a intenção pelo LLM antes do parser determinístico."""
    if llm is None or not turn_id:
        return None
    rendered = render_prompt(state)
    safe_user_text = sanitize_user_text(user_text)
    if not safe_user_text:
        return None
    try:
        decision = llm.invoke_structured(
            turn_id,
            [
                rendered.system_message,
                *_sanitized_history(recent_messages),
                HumanMessage(content=safe_user_text),
            ],
            IntentDecision,
            prompt_version=rendered.prompt_version,
        )
    except IntegrationError:
        return None
    return decision.intent


def _deterministic_intent(user_text: str) -> Intent | None:
    normalized = normalized_text(user_text)
    if any(word in normalized for word in ("cambio", "cotacao", "dolar", "euro")):
        return Intent.EXCHANGE_RATE
    if any(word in normalized for word in ("aumentar", "aumento", "novo limite")):
        return Intent.LIMIT_INCREASE
    if "limite" in normalized:
        return Intent.CREDIT_LIMIT
    return None


def _sanitized_history(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    sanitized: list[BaseMessage] = []
    for message in messages[-6:]:
        if isinstance(message, HumanMessage):
            safe_content = sanitize_user_text(str(message.content))
            if safe_content:
                sanitized.append(HumanMessage(content=safe_content))
        elif isinstance(message, AIMessage) and isinstance(message.content, str):
            sanitized.append(AIMessage(content=message.content))
    return sanitized
