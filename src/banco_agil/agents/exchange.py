"""No de cambio com parser deterministico e cotacao confirmada."""

import re
from datetime import datetime
from zoneinfo import ZoneInfo

from banco_agil.agents._shared import (
    authentication_reply_if_missing,
    end_reply_if_requested,
    format_money,
    normalized_text,
)
from banco_agil.agents.state import ConversationState
from banco_agil.domain.enums import Agent, Intent
from banco_agil.domain.exceptions import IntegrationError
from banco_agil.domain.models import ExchangeRateResult
from banco_agil.services.exchange import ExchangeService
from banco_agil.tools.banking import get_exchange_rate

_BRASILIA_TZ = ZoneInfo("America/Sao_Paulo")

_CURRENCY_NAMES = {
    "dolar": "USD",
    "euro": "EUR",
    "libra": "GBP",
    "real": "BRL",
}

_CURRENCY_FLAGS = {
    "USD": "🇺🇸",
    "EUR": "🇪🇺",
    "GBP": "🇬🇧",
    "BRL": "🇧🇷",
}

_CURRENCY_LABELS = {
    "USD": "dólar",
    "EUR": "euro",
    "GBP": "libra",
    "BRL": "real",
}


def handle_exchange(
    state: ConversationState,
    user_text: str,
    service: ExchangeService,
) -> str:
    """Consulta um par completo sem estimar valores em caso de falha."""
    end_reply = end_reply_if_requested(state, user_text)
    if end_reply is not None:
        return end_reply
    authentication_reply = authentication_reply_if_missing(state)
    if authentication_reply is not None:
        return authentication_reply

    pair = _parse_currency_pair(user_text)
    if pair is None:
        return "Informe o par de moedas desejado, por exemplo USD-BRL."
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
            f"fonte {source}). Posso ajudar em algo mais?"
        )
    return (
        f"{prefix}A cotação {base_currency}-{quote_currency} é {formatted_rate} "
        f"(última atualização às {brasilia_time} horário de Brasília, "
        f"fonte {source}). Posso ajudar em algo mais?"
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
    names = "|".join(_CURRENCY_NAMES)
    named_pair = re.search(rf"\b({names})\s+para\s+({names})\b", normalized)
    if named_pair is not None:
        return (
            _CURRENCY_NAMES[named_pair.group(1)],
            _CURRENCY_NAMES[named_pair.group(2)],
        )
    for alias, currency in _CURRENCY_NAMES.items():
        if alias in normalized.split() and currency != "BRL":
            return currency, "BRL"
    return None
