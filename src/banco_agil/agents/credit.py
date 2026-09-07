"""No de credito para consulta e solicitacao de aumento."""

import re
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation

from langchain_core.messages import BaseMessage

from banco_agil.agents._shared import (
    HELP_REPLY,
    apply_flow_change,
    authentication_reply_if_missing,
    end_reply_if_requested,
    format_money,
    is_help_request,
    normalized_text,
)
from banco_agil.agents.knowledge import explanation_for_pending_step
from banco_agil.agents.state import ConversationState
from banco_agil.agents.understanding import TurnContext, resolve_context
from banco_agil.domain.enums import Agent, CreditRequestStatus, Intent
from banco_agil.domain.exceptions import DomainError, RepositoryError
from banco_agil.domain.models import CreditLimitResult, LimitIncreaseResult
from banco_agil.integrations.llm import StructuredLlm
from banco_agil.services.credit import CreditService
from banco_agil.services.knowledge import KnowledgeService
from banco_agil.tools.banking import get_credit_limit, request_limit_increase


def handle_credit(
    state: ConversationState,
    user_text: str,
    service: CreditService,
    *,
    knowledge: KnowledgeService | None = None,
    context: TurnContext | None = None,
    llm: StructuredLlm | None = None,
    turn_id: str = "",
    recent_messages: Sequence[BaseMessage] = (),
) -> str:
    """Processa consulta, aumento ou reanalise de credito.

    Decisoes continuam deterministicas. O LLM entra quando o texto nao e um
    valor: pode ler desistencia, pedido de outro servico, o valor escrito de
    outro jeito ("uns oito mil") ou redigir a pergunta de esclarecimento.
    """
    context = resolve_context(context, llm, turn_id, recent_messages)
    end_reply = end_reply_if_requested(state, user_text)
    if end_reply is not None:
        return end_reply
    authentication_reply = authentication_reply_if_missing(state)
    if authentication_reply is not None:
        return authentication_reply

    if is_help_request(user_text):
        return HELP_REPLY

    if state.credit_reanalysis_pending and state.requested_limit is not None:
        reanalysis_limit = state.requested_limit
        return _request_increase(state, service, reanalysis_limit)

    if state.intent is Intent.CREDIT_LIMIT:
        result = CreditLimitResult.model_validate(
            get_credit_limit.invoke({"state": state, "service": service})
        )
        state.intent = Intent.UNKNOWN
        state.active_agent = Agent.TRIAGE
        return (
            f"Seu limite atual é R$ {format_money(result.current_limit)}. "
            "Deseja continuar ou encerrar o atendimento?"
        )

    if state.intent is not Intent.LIMIT_INCREASE:
        change = context.flow_change(user_text, state.intent)
        if change is not None:
            return apply_flow_change(state, change)
        return context.clarification(user_text, state.intent) or (
            "Posso consultar seu limite atual ou analisar um pedido de aumento. "
            "Qual dessas opções você prefere?"
        )

    requested_limit = _parse_money(user_text)
    if requested_limit is None and knowledge is not None:
        # Duvida sobre o proprio pedido e respondida antes de virar recusa ou
        # esclarecimento: a pergunta do passo volta no fim da resposta.
        doubt_reply = explanation_for_pending_step(
            user_text,
            knowledge,
            state.authenticated_client,
            "Qual limite total você gostaria de ter?",
            default_topic="score_defines_limit",
        )
        if doubt_reply is not None:
            return doubt_reply
    if requested_limit is None:
        change = context.flow_change(user_text, Intent.LIMIT_INCREASE)
        if change is not None:
            return apply_flow_change(state, change)
        understanding = context.understand(user_text, Intent.LIMIT_INCREASE)
        if understanding is not None and understanding.amount is not None:
            requested_limit = understanding.amount
    if requested_limit is None:
        return _ask_desired_limit(
            state,
            context.clarification(user_text, Intent.LIMIT_INCREASE),
        )
    return _request_increase(state, service, requested_limit)


def _current_limit(state: ConversationState) -> Decimal | None:
    """Le o limite vigente do cliente confiavel do estado, nunca do modelo."""
    client = state.authenticated_client
    return None if client is None else client.credit_limit


def _ask_desired_limit(
    state: ConversationState,
    clarification: str | None,
) -> str:
    """Pergunta o valor desejado dizendo antes quanto o cliente tem hoje.

    Sem o limite atual no texto canonico a redacao final nao poderia cita-lo: a
    guarda de humanizacao so aceita numeros que ja estejam aqui. Com ele, a
    conversa parte do que o cliente tem para o que ele quer ter, em vez de
    perguntar um valor no vazio. O esclarecimento redigido pelo LLM entra como
    a pergunta, quando existe; o fato continua vindo do estado.
    """
    question = clarification or "Qual limite total você gostaria de ter?"
    current_limit = _current_limit(state)
    if current_limit is None:
        return question
    return (
        f"Posso analisar seu aumento. Hoje seu limite é "
        f"R$ {format_money(current_limit)}. {question}"
    )


def _request_increase(
    state: ConversationState,
    service: CreditService,
    requested_limit: Decimal,
) -> str:
    # Lido antes da tool: aprovacao troca o cliente do estado pelo atualizado, e
    # o limite anterior e justamente o que da a dimensao do aumento.
    current_limit = _current_limit(state)
    try:
        result = LimitIncreaseResult.model_validate(
            request_limit_increase.invoke(
                {
                    "new_limit": requested_limit,
                    "state": state,
                    "service": service,
                }
            )
        )
    except DomainError:
        if current_limit is None:
            return (
                "Para solicitar um aumento, o valor precisa ser maior que seu "
                "limite atual. Qual limite total você gostaria de analisar?"
            )
        return (
            f"Hoje seu limite já é R$ {format_money(current_limit)}, então o "
            "aumento precisa ser de um valor acima desse. Qual limite total "
            "você gostaria de ter?"
        )
    except RepositoryError:
        return "Não foi possível processar a solicitação agora. Tente novamente."

    state.credit_reanalysis_pending = False
    if result.status is CreditRequestStatus.REJECTED:
        state.active_agent = Agent.CREDIT_INTERVIEW
        return (
            f"Analisei seu pedido de limite total de R$ "
            f"{format_money(result.requested_limit)}, acima do seu limite atual de "
            f"R$ {format_money(result.current_limit)}, mas ele não pôde ser "
            "aprovado com o score de hoje. Podemos fazer uma entrevista de crédito "
            "para atualizar o score e refazer a análise, sem garantia de aprovação. "
            "Deseja continuar?"
        )

    state.requested_limit = None
    state.intent = Intent.UNKNOWN
    state.active_agent = Agent.TRIAGE
    if current_limit is None or current_limit == result.requested_limit:
        return (
            f"Boa notícia: seu pedido de limite total de R$ "
            f"{format_money(result.requested_limit)} foi aprovado e o limite "
            "cadastrado foi atualizado. Deseja continuar ou encerrar o atendimento?"
        )
    return (
        f"Boa notícia: seu pedido foi aprovado e seu limite cadastrado foi "
        f"atualizado de R$ {format_money(current_limit)} para R$ "
        f"{format_money(result.requested_limit)}. Deseja continuar ou encerrar o "
        "atendimento?"
    )


# "8 mil" e "8k" sao a forma mais comum de dizer um limite em conversa.
_THOUSANDS_PATTERN = re.compile(r"(\d[\d.,]*)\s*(?:mil|k)\b")


def _parse_money(user_text: str) -> Decimal | None:
    thousands = _THOUSANDS_PATTERN.search(normalized_text(user_text))
    if thousands is not None:
        base = _parse_money(thousands.group(1))
        return None if base is None else base * 1000
    match = re.search(r"\d[\d.,]*", user_text)
    if match is None:
        return None
    value = match.group()
    if "," in value:
        value = value.replace(".", "").replace(",", ".")
    elif value.count(".") == 1 and len(value.rsplit(".", maxsplit=1)[1]) == 3:
        value = value.replace(".", "")
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        return None
    return parsed if parsed.is_finite() else None
