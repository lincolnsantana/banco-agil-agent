"""Cliente HTTP tipado para cotacoes da AwesomeAPI."""

from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import cast

import httpx
from pydantic import ValidationError

from banco_agil.domain.exceptions import (
    ExternalServiceUnavailableError,
    IntegrationError,
)
from banco_agil.domain.models import ExchangeQuote
from banco_agil.observability.logging import get_logger

SOURCE_NAME = "AwesomeAPI"
MAX_ATTEMPTS = 2
TRANSIENT_STATUS_CODES = {408, 429}

# A indisponibilidade chega ao cliente como uma frase generica, o que e correto
# para ele e inutil para quem opera: sem registro nao da para distinguir bloqueio
# de rede, limite por IP do provedor e resposta malformada. O par de moedas nao e
# dado pessoal, entao pode ser registrado.
_logger = get_logger("banco_agil.awesomeapi")


class AwesomeApiClient:
    """Consulta a ultima cotacao confirmada para um par de moedas."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 10.0,
        token: str | None = None,
    ) -> None:
        """Configura URL, timeout e token opcional sem realizar requisicao.

        O token viaja como parametro de consulta, como a API exige, e nunca
        entra no log: as mensagens registram a URL base, sem a consulta.
        """
        normalized_url = base_url.rstrip("/")
        if not normalized_url:
            raise ValueError("base URL cannot be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout must be positive")
        self._base_url = normalized_url
        self._timeout_seconds = timeout_seconds
        self._token = token.strip() if token else None

    def get_exchange_rate(
        self,
        base_currency: str,
        quote_currency: str,
    ) -> ExchangeQuote:
        """Retorna uma cotacao validada da API.

        Raises:
            ValueError: Se o par de moedas for invalido.
            IntegrationError: Se a resposta externa for invalida ou rejeitada.
            ExternalServiceUnavailableError: Se duas tentativas transitorias falharem.
        """
        base = _normalize_currency(base_currency)
        quote = _normalize_currency(quote_currency)
        if base == quote:
            raise ValueError("base and quote currencies must be different")

        url = f"{self._base_url}/json/last/{base}-{quote}"
        response = self._request_with_retry(url, f"{base}-{quote}")
        return _parse_quote(response, base, quote)

    def _request_with_retry(self, url: str, pair: str) -> httpx.Response:
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = httpx.get(
                    url,
                    params={"token": self._token} if self._token else None,
                    timeout=self._timeout_seconds,
                )
            except httpx.RequestError as error:
                _logger.warning(
                    "exchange request failed: pair=%s attempt=%d error=%s url=%s",
                    pair,
                    attempt + 1,
                    type(error).__name__,
                    self._base_url,
                    extra={
                        "audit": {
                            "pair": pair,
                            "attempt": attempt + 1,
                            "error": type(error).__name__,
                            "base_url": self._base_url,
                        }
                    },
                )
                if attempt + 1 == MAX_ATTEMPTS:
                    raise ExternalServiceUnavailableError(
                        "exchange rate provider is unavailable"
                    ) from error
                continue

            if (
                response.status_code in TRANSIENT_STATUS_CODES
                or response.status_code >= 500
            ):
                _logger.warning(
                    "exchange provider returned a transient status: "
                    "pair=%s attempt=%d status=%d body=%s",
                    pair,
                    attempt + 1,
                    response.status_code,
                    _log_excerpt(response),
                    extra={
                        "audit": {
                            "pair": pair,
                            "attempt": attempt + 1,
                            "status": response.status_code,
                        }
                    },
                )
                if attempt + 1 == MAX_ATTEMPTS:
                    raise ExternalServiceUnavailableError(
                        "exchange rate provider is unavailable"
                    )
                continue
            if not 200 <= response.status_code < 300:
                _logger.warning(
                    "exchange request was rejected: pair=%s status=%d body=%s",
                    pair,
                    response.status_code,
                    _log_excerpt(response),
                    extra={"audit": {"pair": pair, "status": response.status_code}},
                )
                raise IntegrationError("exchange rate request was rejected")
            return response

        raise ExternalServiceUnavailableError("exchange rate provider is unavailable")


def _log_excerpt(response: httpx.Response) -> str:
    """Primeiros caracteres do corpo, para o log dizer o motivo da recusa.

    Provedor de cotacao nao devolve dado pessoal, e o trecho e curto de
    proposito: serve para distinguir limite de uso de erro de rota, sem
    despejar a resposta inteira no log.
    """
    try:
        return " ".join(response.text.split())[:120]
    except Exception:  # pragma: no cover - corpo ilegivel nao pode quebrar o log
        return "<indisponivel>"


def _normalize_currency(currency: str) -> str:
    normalized = currency.strip().upper()
    if len(normalized) != 3 or not normalized.isascii() or not normalized.isalpha():
        raise ValueError("currency must contain three ASCII letters")
    return normalized


def _parse_quote(
    response: httpx.Response,
    base_currency: str,
    quote_currency: str,
) -> ExchangeQuote:
    try:
        payload: object = response.json()
        if not isinstance(payload, dict):
            raise ValueError("response root must be an object")
        payload_mapping = cast(Mapping[object, object], payload)
        quote_data = payload_mapping.get(f"{base_currency}{quote_currency}")
        if not isinstance(quote_data, dict):
            raise ValueError("currency pair is missing")
        quote_mapping = cast(Mapping[object, object], quote_data)

        code = _required_string(quote_mapping, "code")
        codein = _required_string(quote_mapping, "codein")
        bid = _required_string(quote_mapping, "bid")
        timestamp = _required_string(quote_mapping, "timestamp")
        if code != base_currency or codein != quote_currency:
            raise ValueError("response currency pair does not match request")

        rate = Decimal(bid)
        unix_timestamp = int(timestamp)
        if unix_timestamp <= 0:
            raise ValueError("quote timestamp must be positive")
        quoted_at = datetime.fromtimestamp(unix_timestamp, tz=UTC)
        return ExchangeQuote(
            base_currency=base_currency,
            quote_currency=quote_currency,
            rate=rate,
            source=SOURCE_NAME,
            quoted_at=quoted_at,
        )
    except (
        InvalidOperation,
        OSError,
        OverflowError,
        TypeError,
        ValueError,
        ValidationError,
    ) as error:
        raise IntegrationError("invalid exchange rate response") from error


def _required_string(data: Mapping[object, object], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str):
        raise ValueError("required quote fields are missing")
    return value
