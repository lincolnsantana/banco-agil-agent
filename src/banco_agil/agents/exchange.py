"""No de cambio com parser deterministico e cotacao confirmada."""

import re
from collections.abc import Sequence
from datetime import datetime
from zoneinfo import ZoneInfo

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
from banco_agil.domain.enums import Agent, Intent
from banco_agil.domain.exceptions import IntegrationError
from banco_agil.domain.models import ExchangeRateResult
from banco_agil.integrations.llm import StructuredLlm
from banco_agil.services.exchange import ExchangeService
from banco_agil.tools.banking import get_exchange_rate

_BRASILIA_TZ = ZoneInfo("America/Sao_Paulo")

_CURRENCY_NAMES = {
    "dolar": "USD",
    "dolar americano": "USD",
    "euro": "EUR",
    "libra": "GBP",
    "libra esterlina": "GBP",
    "real": "BRL",
    "peso argentino": "ARS",
    "iene": "JPY",
    "franco suico": "CHF",
    "dolar canadense": "CAD",
    "dolar australiano": "AUD",
    "yuan": "CNY",
    "bitcoin": "BTC",
}

_CURRENCY_LIST_REQUESTS = (
    "quais moedas",
    "que moedas",
    "moedas disponiveis",
    "moedas aceita",
    "moedas suporta",
    "moedas consulta",
    "moedas posso",
    "moedas que",
    "tipos de moeda",
    "sobre as moedas",
)

_CURRENCY_LIST_REPLY = (
    "Posso consultar dólar americano, euro, libra esterlina, peso argentino, "
    "iene, franco suíço, dólar canadense, dólar australiano, yuan e bitcoin "
    "em reais. Você pode informar só o nome, como 'cotação do euro', ou um "
    "par específico, como EUR-USD. Qual cotação deseja consultar?"
)

_CURRENCY_FLAGS = {
    "USD": "🇺🇸",
    "EUR": "🇪🇺",
    "GBP": "🇬🇧",
    "BRL": "🇧🇷",
    "ARS": "🇦🇷",
    "JPY": "🇯🇵",
    "CHF": "🇨🇭",
    "CAD": "🇨🇦",
    "AUD": "🇦🇺",
    "CNY": "🇨🇳",
}

_CURRENCY_LABELS = {
    "USD": "dólar",
    "EUR": "euro",
    "GBP": "libra",
    "BRL": "real",
    "ARS": "peso argentino",
    "JPY": "iene",
    "CHF": "franco suíço",
    "CAD": "dólar canadense",
    "AUD": "dólar australiano",
    "CNY": "yuan",
    "BTC": "bitcoin",
}


def handle_exchange(
    state: ConversationState,
    user_text: str,
    service: ExchangeService,
    *,
    context: TurnContext | None = None,
    llm: StructuredLlm | None = None,
    turn_id: str = "",
    recent_messages: Sequence[BaseMessage] = (),
) -> str:
    """Consulta um par completo sem estimar valores em caso de falha.

    Sem moeda reconhecida pelo parser, o LLM pode ler desistencia, pedido de
    outro servico ou o par dito de outro jeito; a cotacao continua vindo so da
    tool.
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

    normalized = normalized_text(user_text)
    if any(request in normalized for request in _CURRENCY_LIST_REQUESTS):
        return _CURRENCY_LIST_REPLY

    pair = _parse_currency_pair(user_text)
    if pair is None:
        change = context.flow_change(user_text, Intent.EXCHANGE_RATE)
        if change is not None:
            return apply_flow_change(state, change)
        understanding = context.understand(user_text, Intent.EXCHANGE_RATE)
        if understanding is not None:
            pair = understanding.currency_pair
    if pair is None:
        return context.clarification(user_text, Intent.EXCHANGE_RATE) or (
            "Qual moeda você quer consultar? Pode dizer apenas o nome, como "
            "dólar ou euro, ou informar um par, como EUR-USD."
        )
    if pair[0] == pair[1]:
        return "Informe duas moedas diferentes para consultar a cotação."

    try:
        result = ExchangeRateResult.model_validate(
            get_exchange_rate.invoke(
                {
                    "base_currency": pair[0],
                    "quote_currency": pair[1],
                    "state": state,
                    "service": service,
                }
            )
        )
    except IntegrationError:
        return "A cotação está indisponível no momento. Tente novamente mais tarde."

    quote = result.quote
    state.intent = Intent.UNKNOWN
    state.active_agent = Agent.TRIAGE
    return _format_quote(
        quote.base_currency,
        quote.quote_currency,
        format_money(quote.rate),
        quote.source,
        _brasilia_time(quote.quoted_at),
    )


def _brasilia_time(quoted_at: datetime) -> str:
    """Converte o instante da cotação para HH:MM no horário de Brasília."""
    return quoted_at.astimezone(_BRASILIA_TZ).strftime("%H:%M")


def _format_quote(
    base_currency: str,
    quote_currency: str,
    formatted_rate: str,
    source: str,
    brasilia_time: str,
) -> str:
    """Monta o canônico amigável; o Groq só reescreve via `humanize_reply`."""
    flag = _CURRENCY_FLAGS.get(base_currency, "")
    prefix = f"{flag} " if flag else ""
    if quote_currency == "BRL":
        label = _CURRENCY_LABELS.get(base_currency, base_currency)
        return (
            f"{prefix}O {label} está em R$ {formatted_rate} "
            f"(última atualização às {brasilia_time} horário de Brasília, "
            f"fonte {source}). Deseja continuar ou encerrar o atendimento?"
        )
    return (
        f"{prefix}A cotação {base_currency}-{quote_currency} é {formatted_rate} "
        f"(última atualização às {brasilia_time} horário de Brasília, "
        f"fonte {source}). Deseja continuar ou encerrar o atendimento?"
    )


def _parse_currency_pair(user_text: str) -> tuple[str, str] | None:
    explicit_pair = re.search(
        r"\b([A-Za-z]{3})\s*(?:-|/|\bpara\b)\s*([A-Za-z]{3})\b",
        user_text,
        flags=re.IGNORECASE,
    )
    if explicit_pair is not None:
        return explicit_pair.group(1).upper(), explicit_pair.group(2).upper()

    normalized = normalized_text(user_text)
    names = "|".join(
        re.escape(alias) for alias in sorted(_CURRENCY_NAMES, key=len, reverse=True)
    )
    named_pair = re.search(rf"\b({names})\s+(?:para|em)\s+({names})\b", normalized)
    if named_pair is not None:
        return (
            _CURRENCY_NAMES[named_pair.group(1)],
            _CURRENCY_NAMES[named_pair.group(2)],
        )
    for alias in sorted(_CURRENCY_NAMES, key=len, reverse=True):
        currency = _CURRENCY_NAMES[alias]
        if re.search(rf"\b{re.escape(alias)}\b", normalized) and currency != "BRL":
            return currency, "BRL"
    # Codigo solto so em maiusculas: em minusculas, "nao" ou "vou" virariam
    # moeda e iriam para a API em vez de serem lidos como recusa.
    standalone_code = re.search(r"\b([A-Z]{3})\b", user_text)
    if standalone_code is not None and standalone_code.group(1) != "BRL":
        return standalone_code.group(1), "BRL"
    return None
