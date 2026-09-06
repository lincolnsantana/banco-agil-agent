"""Nó de triagem determinístico com fallback de intenção pelo LLM."""

import re
from collections.abc import Sequence

from langchain_core.messages import BaseMessage

from banco_agil.agents._shared import (
    GREETING_REPLY,
    HANDOFF_REPLY,
    HELP_REPLY,
    RESUMABLE_INTENTS,
    agent_for_intent,
    apply_flow_change,
    classify_banking_request,
    detect_howto_topic,
    detects_information_question,
    end_conversation,
    end_reply_if_requested,
    is_greeting,
    is_help_request,
    parse_flow_answer,
)
from banco_agil.agents.state import ConversationState
from banco_agil.agents.understanding import TurnContext, resolve_context
from banco_agil.domain.enums import Agent, EndReason, Intent
from banco_agil.domain.exceptions import RepositoryError
from banco_agil.domain.models import AuthenticationResult, CpfValidationResult
from banco_agil.integrations.llm import StructuredLlm
from banco_agil.services.authentication import AuthenticationService
from banco_agil.tools.banking import authenticate_client, validate_client_cpf


def handle_triage(
    state: ConversationState,
    user_text: str,
    service: AuthenticationService,
    *,
    context: TurnContext | None = None,
    llm: StructuredLlm | None = None,
    turn_id: str = "",
    recent_messages: Sequence[BaseMessage] = (),
) -> str:
    """Processa um turno de triagem e atualiza o estado confiavel."""
    context = resolve_context(context, llm, turn_id, recent_messages)
    end_reply = end_reply_if_requested(state, user_text)
    if end_reply is not None:
        return end_reply

    if not state.authenticated:
        _remember_requested_intent(state, user_text)
        return _handle_authentication(state, user_text, service)

    if state.pending_flow is not None and not _supersedes_pending_flow(
        user_text, state.pending_flow
    ):
        return _handle_flow_answer(state, user_text, context)
    if state.pending_flow is not None:
        # Pedido novo no lugar da confirmacao: a oferta anterior caduca, em vez
        # de repetir "nao consegui confirmar" enquanto o cliente muda de assunto.
        state.pending_flow = None

    topic = detect_howto_topic(user_text)
    if topic is not None:
        return _handle_howto(state, topic)

    if is_greeting(user_text):
        # Cumprimento pede cumprimento de volta, nao um menu seco.
        state.intent = Intent.UNKNOWN
        state.active_agent = Agent.TRIAGE
        return GREETING_REPLY

    if is_help_request(user_text):
        state.intent = Intent.UNKNOWN
        state.active_agent = Agent.TRIAGE
        return HELP_REPLY

    if detects_information_question(user_text):
        # Vem depois do howto e da ajuda, que ja tem destino proprio, e antes
        # da classificacao por acao, que transformaria a duvida em operacao.
        state.intent = Intent.INFORMATION
        state.active_agent = Agent.KNOWLEDGE
        return "Certo. Vou explicar."

    intent = _deterministic_intent(user_text)
    understanding = None
    if intent is None:
        understanding = context.understand(user_text, Intent.UNKNOWN)
        intent = understanding.intent if understanding is not None else None

    if intent is Intent.HELP:
        state.intent = Intent.UNKNOWN
        state.active_agent = Agent.TRIAGE
        return HELP_REPLY

    if intent is Intent.INFORMATION:
        # O LLM reconheceu uma duvida que o parser nao pegou; o Conhecimento
        # reaproveita a leitura para achar o topico no catalogo.
        state.intent = Intent.INFORMATION
        state.active_agent = Agent.KNOWLEDGE
        return "Certo. Vou explicar."

    if intent not in RESUMABLE_INTENTS:
        clarification = (
            understanding.clarification if understanding is not None else None
        )
        return clarification or (
            "Posso ajudar com limite de crédito ou cotação de moedas. O que deseja?"
        )

    state.intent = intent
    state.active_agent = agent_for_intent(intent)
    return HANDOFF_REPLY


def _handle_howto(state: ConversationState, topic: Intent) -> str:
    """Explica o fluxo e confirma antes de iniciar entrevista ou câmbio."""
    if topic is Intent.CREDIT_LIMIT:
        state.intent = Intent.CREDIT_LIMIT
        state.active_agent = Agent.CREDIT
        return HANDOFF_REPLY
    if topic is Intent.CREDIT_INTERVIEW:
        # Perguntar como melhorar o score ja e pedir a entrevista: o
        # especialista abre explicando e faz a primeira pergunta no mesmo turno.
        state.intent = Intent.CREDIT_INTERVIEW
        state.active_agent = Agent.CREDIT_INTERVIEW
        return HANDOFF_REPLY
    if topic is Intent.LIMIT_INCREASE:
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


def _supersedes_pending_flow(user_text: str, pending: Intent | None) -> bool:
    """Indica que o cliente trocou de assunto em vez de confirmar a oferta.

    "quero" sozinho confirma a oferta, mas "quero aumentar meu limite" nomeia
    outro servico e tambem casa com o parser de afirmativa. Por isso o pedido
    so vence quando aponta para um servico diferente do pendente. Resposta
    vaga como "nao sei" nao entra aqui: repetir a pergunta ainda e o certo.
    """
    requested = classify_banking_request(user_text)
    if requested is not None and requested is not pending:
        return True
    if parse_flow_answer(user_text) is not None:
        return False
    return detects_information_question(user_text)


def _handle_flow_answer(
    state: ConversationState,
    user_text: str,
    context: TurnContext,
) -> str:
    """Confirma o fluxo pendente com resposta ampla ou repete a pergunta.

    Resposta que o parser nao le vai ao LLM, que pode reconhecer recusa ou um
    pedido diferente da oferta; sem isso, a pergunta e repetida.
    """
    target = state.pending_flow
    answer = parse_flow_answer(user_text)
    if answer is None and target is not None:
        understanding = context.understand(user_text, target)
        change = (
            understanding.flow_change(target) if understanding is not None else None
        )
        if change is not None and not change.declines_current:
            return apply_flow_change(state, change)
        if change is not None:
            answer = False
    if answer is True and target is not None:
        state.pending_flow = None
        state.intent = target
        state.active_agent = agent_for_intent(target)
        return HANDOFF_REPLY
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


def _remember_requested_intent(state: ConversationState, user_text: str) -> None:
    """Guarda o pedido feito antes da autenticacao, preservando o primeiro.

    CPF e nascimento nao carregam intencao, entao as respostas de autenticacao
    passam por aqui sem sobrescrever o que o cliente pediu na abertura.
    """
    if state.deferred_intent is not None:
        return
    intent = _deterministic_intent(user_text)
    if intent in RESUMABLE_INTENTS:
        state.deferred_intent = intent
        state.deferred_request = user_text


def _resume_requested_intent(state: ConversationState) -> str:
    """Retoma o pedido anterior a autenticacao em vez de perguntar de novo.

    A resposta daqui e substituida pela do especialista quando ha pedido a
    retomar: o grafo segue para ele no mesmo turno.
    """
    deferred = state.deferred_intent
    if deferred is None:
        state.deferred_request = None
        return "Dados confirmados. Como posso ajudar hoje?"
    # O texto fica no estado de proposito: o grafo o entrega ao especialista
    # neste mesmo turno, no lugar da data de nascimento, e so entao o descarta.
    state.deferred_intent = None
    state.intent = deferred
    state.active_agent = agent_for_intent(deferred)
    return "Dados confirmados. Vou retomar seu pedido."


def _deterministic_intent(user_text: str) -> Intent | None:
    """Reaproveita o classificador que casa acao e substantivo do pedido."""
    return classify_banking_request(user_text)
