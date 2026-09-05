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
from banco_agil.agents.state import ConversationState
from banco_agil.agents.understanding import TurnContext, resolve_context
from banco_agil.domain.enums import Agent, CreditRequestStatus, Intent
from banco_agil.domain.exceptions import DomainError, RepositoryError
from banco_agil.domain.models import CreditLimitResult, LimitIncreaseResult
from banco_agil.integrations.llm import StructuredLlm
from banco_agil.services.credit import CreditService
from banco_agil.tools.banking import get_credit_limit, request_limit_increase


def handle_credit(
    state: ConversationState,
    user_text: str,
    service: CreditService,
    *,
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
    if requested_limit is None:
        change = context.flow_change(user_text, Intent.LIMIT_INCREASE)
        if change is not None:
            return apply_flow_change(state, change)
        understanding = context.understand(user_text, Intent.LIMIT_INCREASE)
        if understanding is not None and understanding.amount is not None:
            requested_limit = understanding.amount
    if requested_limit is None:
        return context.clarification(user_text, Intent.LIMIT_INCREASE) or (
            "Claro, posso analisar o aumento com você. Qual é o limite total que "
            "gostaria de ter? Por exemplo: R$ 4.000,00."
        )
    return _request_increase(state, service, requested_limit)


def _request_increase(
    state: ConversationState,
    service: CreditService,
    requested_limit: Decimal,
) -> str:
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
        return (
            "Para solicitar um aumento, o valor precisa ser maior que seu limite "
            "atual. Qual limite total você gostaria de analisar?"
        )
    except RepositoryError:
        return "Não foi possível processar a solicitação agora. Tente novamente."

    state.credit_reanalysis_pending = False
    if result.status is CreditRequestStatus.REJECTED:
        state.active_agent = Agent.CREDIT_INTERVIEW
        return (
            f"Analisei seu pedido de limite total de R$ "
            f"{format_money(result.requested_limit)}, mas ele não pôde ser aprovado "
            "com o score atual. Se quiser, podemos fazer uma entrevista de crédito "
            "para atualizar o score e realizar uma nova análise, sem garantia de "
            "aprovação. Deseja continuar?"
        )

    state.requested_limit = None
    state.intent = Intent.UNKNOWN
    state.active_agent = Agent.TRIAGE
    return (
        f"Boa notícia: seu pedido de limite total de R$ "
        f"{format_money(result.requested_limit)} foi aprovado e o limite cadastrado "
        "foi atualizado. Deseja continuar ou encerrar o atendimento?"
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
