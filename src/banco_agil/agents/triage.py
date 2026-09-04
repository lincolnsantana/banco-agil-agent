"""Nó de triagem determinístico com fallback de intenção pelo LLM."""

import re
from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel

from banco_agil.agents._shared import (
    HELP_REPLY,
    detect_howto_topic,
    end_conversation,
    end_reply_if_requested,
    is_help_request,
    normalized_text,
    parse_flow_answer,
    sanitize_user_text,
)
from banco_agil.agents.state import ConversationState
from banco_agil.domain.enums import Agent, EndReason, Intent
from banco_agil.domain.exceptions import IntegrationError, RepositoryError
from banco_agil.domain.models import AuthenticationResult, CpfValidationResult
from banco_agil.integrations.llm import StructuredLlm
from banco_agil.prompts.renderer import render_prompt
from banco_agil.services.authentication import AuthenticationService
from banco_agil.tools.banking import authenticate_client, validate_client_cpf


class IntentDecision(BaseModel):
    """Intenção bancária classificada sem permitir escolha direta de agente."""

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

    if state.pending_flow is not None:
        return _handle_flow_answer(state, user_text)

    topic = detect_howto_topic(user_text)
    if topic is not None:
        return _handle_howto(state, topic)

    if is_help_request(user_text):
        state.intent = Intent.UNKNOWN
        state.active_agent = Agent.TRIAGE
        return HELP_REPLY

    intent = _deterministic_intent(user_text)
    if intent is None:
        intent = _llm_intent(state, user_text, llm, turn_id, recent_messages)

    if intent is Intent.HELP:
        state.intent = Intent.UNKNOWN
        state.active_agent = Agent.TRIAGE
        return HELP_REPLY

    if intent not in {
        Intent.CREDIT_LIMIT,
        Intent.LIMIT_INCREASE,
        Intent.CREDIT_INTERVIEW,
        Intent.EXCHANGE_RATE,
    }:
        return "Posso ajudar com limite de crédito ou cotação de moedas. O que deseja?"

    state.intent = intent
    if intent is Intent.EXCHANGE_RATE:
        state.active_agent = Agent.EXCHANGE
    elif intent is Intent.CREDIT_INTERVIEW:
        state.active_agent = Agent.CREDIT_INTERVIEW
    else:
        state.active_agent = Agent.CREDIT
    return "Certo. Vou prosseguir com sua solicitação."


def _handle_howto(state: ConversationState, topic: Intent) -> str:
    """Explica o fluxo e confirma antes de iniciar aumento ou câmbio."""
    if topic is Intent.CREDIT_LIMIT:
        state.intent = Intent.CREDIT_LIMIT
        state.active_agent = Agent.CREDIT
        return "Certo. Vou prosseguir com sua solicitação."
    if topic is Intent.CREDIT_INTERVIEW:
        state.intent = Intent.CREDIT_INTERVIEW
        state.active_agent = Agent.CREDIT_INTERVIEW
        return (
            "Na entrevista, faço 5 perguntas — renda, emprego, despesas, "
            "dependentes e dívidas —, recalculo seu score e, se houver um "
            "pedido rejeitado, reanaliso na hora. Deseja realizar a "
            "entrevista de crédito? Responda sim ou não."
        )
    if topic is Intent.EXCHANGE_RATE:
        explanation = (
            "Para consultar, me diga o par de moedas (por exemplo: USD-BRL) "
            "ou pergunte direto o valor do dólar hoje."
        )
    else:
        explanation = (
            "Para aumentar, você me informa o novo limite total desejado; eu "
            "registro o pedido e avalio na hora pelo seu score."
        )
    state.pending_flow = topic
    state.intent = Intent.UNKNOWN
    state.active_agent = Agent.TRIAGE
    return f"{explanation} Quer realizar agora? Responda sim ou não."


def _handle_flow_answer(state: ConversationState, user_text: str) -> str:
    """Confirma o fluxo pendente com resposta ampla ou repete a pergunta."""
    target = state.pending_flow
    answer = parse_flow_answer(user_text)
    if answer is True and target is not None:
        state.pending_flow = None
        state.intent = target
        state.active_agent = (
            Agent.EXCHANGE if target is Intent.EXCHANGE_RATE else Agent.CREDIT
        )
        return "Certo. Vou prosseguir com sua solicitação."
    if answer is False:
        state.pending_flow = None
        state.intent = Intent.UNKNOWN
        state.active_agent = Agent.TRIAGE
        return (
            "Tudo bem. Se mudar de ideia, é só pedir. Posso ajudar com limite "
            "de crédito ou cotação de moedas. O que deseja?"
        )
    return "Não entendi. Deseja realizar? Responda sim ou não."


def _handle_authentication(
    state: ConversationState,
    user_text: str,
    service: AuthenticationService,
) -> str:
    if state.pending_cpf is None:
        if not re.fullmatch(r"[\d.\-\s]+", user_text.strip()):
            return (
                "Antes de continuar, precisamos validar alguns dados para proteger "
                "seu atendimento. Por favor, informe seu CPF com 11 dígitos."
            )
        try:
            cpf_result = CpfValidationResult.model_validate(
                validate_client_cpf.invoke(
                    {"cpf": user_text, "state": state, "service": service}
                )
            )
        except RepositoryError:
            return "Não foi possível validar o CPF agora. Tente novamente mais tarde."
        if cpf_result.should_end:
            end_conversation(state, EndReason.AUTHENTICATION_FAILURES)
            return (
                "Não foi possível validar o CPF após três tentativas. "
                "Atendimento encerrado."
            )
        if not cpf_result.valid:
            return (
                "O CPF informado é inválido. Digite novamente seu CPF com 11 dígitos."
            )
        return (
            "CPF localizado. Para concluir a autenticação, informe sua data de "
            "nascimento no formato DD/MM/AAAA."
        )

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
    return (
        "Não foi possível validar os dados informados. Vamos tentar novamente: "
        "informe seu CPF com 11 dígitos."
    )


def _deterministic_intent(user_text: str) -> Intent | None:
    normalized = normalized_text(user_text)
    if any(word in normalized for word in ("cambio", "cotacao", "dolar", "euro")):
        return Intent.EXCHANGE_RATE
    if any(
        term in normalized
        for term in ("entrevista", "score", "pontuacao", "pontos de credito")
    ):
        return Intent.CREDIT_INTERVIEW
    if any(term in normalized for term in ("aumentar", "aumento", "novo limite")):
        return Intent.LIMIT_INCREASE
    if "limite" in normalized and any(
        term in normalized for term in ("alterar", "ajustar", "modificar", "mudar")
    ):
        return Intent.LIMIT_INCREASE
    if "limite" in normalized:
        return Intent.CREDIT_LIMIT
    return None


def _llm_intent(
    state: ConversationState,
    user_text: str,
    llm: StructuredLlm | None,
    turn_id: str,
    recent_messages: Sequence[BaseMessage],
) -> Intent | None:
    """Classifica somente texto pós-autenticação não resolvido pelo parser."""
    if llm is None or not turn_id:
        return None
    safe_user_text = sanitize_user_text(user_text)
    if not safe_user_text:
        return None
    rendered = render_prompt(state)
    messages: list[BaseMessage] = [
        rendered.system_message,
        *_sanitized_history(recent_messages),
        HumanMessage(content=safe_user_text),
    ]
    try:
        decision = llm.invoke_structured(
            turn_id,
            messages,
            IntentDecision,
            prompt_version=rendered.prompt_version,
        )
    except IntegrationError:
        return None
    return decision.intent


def _sanitized_history(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    sanitized: list[BaseMessage] = []
    for message in messages[-5:]:
        if not isinstance(message, (HumanMessage, AIMessage)):
            continue
        safe_content = sanitize_user_text(str(message.content))
        if not safe_content:
            continue
        message_type = AIMessage if isinstance(message, AIMessage) else HumanMessage
        sanitized.append(message_type(content=safe_content))
    return sanitized
