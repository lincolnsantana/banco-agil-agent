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
        _remember_requested_intent(state, user_text)
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
    state.active_agent = agent_for_intent(intent)
    return "Certo. Vou prosseguir com sua solicitação."


def _handle_howto(state: ConversationState, topic: Intent) -> str:
    """Explica o fluxo e confirma antes de iniciar entrevista ou câmbio."""
    if topic is Intent.CREDIT_LIMIT:
        state.intent = Intent.CREDIT_LIMIT
        state.active_agent = Agent.CREDIT
        return "Certo. Vou prosseguir com sua solicitação."
    if topic in {Intent.CREDIT_INTERVIEW, Intent.LIMIT_INCREASE}:
        state.pending_flow = Intent.CREDIT_INTERVIEW
        state.intent = Intent.UNKNOWN
        state.active_agent = Agent.TRIAGE
        return (
            "Para verificar se é possível melhorar seu score e aumentar seu "
            "limite, posso fazer uma entrevista de crédito. São 5 perguntas "
            "sobre renda, emprego, despesas, dependentes e dívidas; com as "
            "respostas, recalculo seu score, sem garantir aprovação. Quer "
            "realizar a entrevista agora?"
        )
    if topic is Intent.EXCHANGE_RATE:
        explanation = (
            "Para consultar, basta dizer o nome da moeda, como dólar ou euro. "
            "Se quiser outra conversão, também pode informar um par, como EUR-USD."
        )
    state.pending_flow = topic
    state.intent = Intent.UNKNOWN
    state.active_agent = Agent.TRIAGE
    return f"{explanation} Quer que eu faça isso agora?"


def _handle_flow_answer(state: ConversationState, user_text: str) -> str:
    """Confirma o fluxo pendente com resposta ampla ou repete a pergunta."""
    target = state.pending_flow
    answer = parse_flow_answer(user_text)
    if answer is True and target is not None:
        state.pending_flow = None
        state.intent = target
        state.active_agent = agent_for_intent(target)
        return "Certo. Vou prosseguir com sua solicitação."
    if answer is False:
        state.pending_flow = None
        state.intent = Intent.UNKNOWN
        state.active_agent = Agent.TRIAGE
        return (
            "Tudo bem. Se mudar de ideia, é só pedir. Posso ajudar com limite "
            "de crédito ou cotação de moedas. O que deseja?"
        )
    return "Não consegui confirmar. Quer que eu faça isso agora?"


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
        return _resume_requested_intent(state)

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


# Pedidos que fazem sentido retomar sozinhos assim que a autenticacao conclui.
_RESUMABLE_INTENTS = frozenset(
    {
        Intent.CREDIT_LIMIT,
        Intent.LIMIT_INCREASE,
        Intent.CREDIT_INTERVIEW,
        Intent.EXCHANGE_RATE,
    }
)


def agent_for_intent(intent: Intent) -> Agent:
    """Traduz a intencao de atendimento no especialista responsavel."""
    if intent is Intent.EXCHANGE_RATE:
        return Agent.EXCHANGE
    if intent is Intent.CREDIT_INTERVIEW:
        return Agent.CREDIT_INTERVIEW
    return Agent.CREDIT


def _remember_requested_intent(state: ConversationState, user_text: str) -> None:
    """Guarda o pedido feito antes da autenticacao, preservando o primeiro.

    CPF e nascimento nao carregam intencao, entao as respostas de autenticacao
    passam por aqui sem sobrescrever o que o cliente pediu na abertura.
    """
    if state.deferred_intent is not None:
        return
    intent = _deterministic_intent(user_text)
    if intent in _RESUMABLE_INTENTS:
        state.deferred_intent = intent


def _resume_requested_intent(state: ConversationState) -> str:
    """Retoma o pedido anterior a autenticacao em vez de perguntar de novo.

    A resposta daqui e substituida pela do especialista quando ha pedido a
    retomar: o grafo segue para ele no mesmo turno.
    """
    deferred = state.deferred_intent
    if deferred is None:
        return "Dados confirmados. Como posso ajudar hoje?"
    state.deferred_intent = None
    state.intent = deferred
    state.active_agent = agent_for_intent(deferred)
    return "Dados confirmados. Vou retomar seu pedido."


def _deterministic_intent(user_text: str) -> Intent | None:
    normalized = normalized_text(user_text)
    if any(
        word in normalized
        for word in ("cambio", "cotacao", "dolar", "euro", "moeda", "moedas")
    ):
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
