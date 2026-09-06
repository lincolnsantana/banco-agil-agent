"""Testes offline da integracao com a AwesomeAPI."""

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal

import httpx
import pytest
import respx

from banco_agil.agents.state import ConversationState
from banco_agil.domain.exceptions import (
    AuthorizationError,
    ExternalServiceUnavailableError,
    IntegrationError,
)
from banco_agil.domain.models import Client, ExchangeQuote
from banco_agil.integrations.awesomeapi import AwesomeApiClient
from banco_agil.services.exchange import ExchangeService

BASE_URL = "https://economia.example.test"
QUOTE_URL = f"{BASE_URL}/json/last/USD-BRL"
TIMESTAMP = "1618315045"


def quote_payload(
    *,
    code: str = "USD",
    codein: str = "BRL",
    bid: str = "5.7276",
    timestamp: str = TIMESTAMP,
) -> dict[str, dict[str, str]]:
    """Monta uma resposta ficticia no contrato oficial da API."""
    return {
        f"{code}{codein}": {
            "code": code,
            "codein": codein,
            "bid": bid,
            "timestamp": timestamp,
        }
    }


@respx.mock
def test_returns_typed_quote_from_confirmed_response() -> None:
    route = respx.get(QUOTE_URL).mock(
        return_value=httpx.Response(200, json=quote_payload())
    )
    client = AwesomeApiClient(BASE_URL, timeout_seconds=2.0)

    quote = client.get_exchange_rate("usd", "brl")

    assert route.call_count == 1
    assert quote.base_currency == "USD"
    assert quote.quote_currency == "BRL"
    assert quote.rate == Decimal("5.7276")
    assert quote.source == "AwesomeAPI"
    assert quote.quoted_at == datetime.fromtimestamp(int(TIMESTAMP), tz=UTC)


@respx.mock
def test_supports_another_currency_pair() -> None:
    url = f"{BASE_URL}/json/last/EUR-BRL"
    respx.get(url).mock(
        return_value=httpx.Response(
            200,
            json=quote_payload(code="EUR", bid="6.8195"),
        )
    )
    client = AwesomeApiClient(BASE_URL)

    quote = client.get_exchange_rate("EUR", "BRL")

    assert quote.rate == Decimal("6.8195")


@pytest.mark.parametrize("currency", ["US", "US1", "DÓL", ""])
def test_rejects_invalid_currency_before_http(currency: str) -> None:
    client = AwesomeApiClient(BASE_URL)

    with pytest.raises(ValueError, match="currency"):
        client.get_exchange_rate(currency, "BRL")


def test_rejects_equal_currency_pair() -> None:
    client = AwesomeApiClient(BASE_URL)

    with pytest.raises(ValueError, match="different"):
        client.get_exchange_rate("USD", "USD")


@respx.mock
def test_retries_once_after_timeout_then_succeeds() -> None:
    route = respx.get(QUOTE_URL).mock(
        side_effect=[
            httpx.ReadTimeout("simulated timeout"),
            httpx.Response(200, json=quote_payload()),
        ]
    )
    client = AwesomeApiClient(BASE_URL)

    quote = client.get_exchange_rate("USD", "BRL")

    assert quote.rate == Decimal("5.7276")
    assert route.call_count == 2


@respx.mock
def test_retries_once_after_server_error_then_succeeds() -> None:
    route = respx.get(QUOTE_URL).mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, json=quote_payload()),
        ]
    )
    client = AwesomeApiClient(BASE_URL)

    client.get_exchange_rate("USD", "BRL")

    assert route.call_count == 2


@pytest.mark.parametrize(
    "error",
    [httpx.ReadTimeout("simulated timeout"), httpx.ConnectError("offline")],
)
@respx.mock
def test_persistent_network_failure_is_controlled(error: httpx.RequestError) -> None:
    route = respx.get(QUOTE_URL).mock(side_effect=error)
    client = AwesomeApiClient(BASE_URL)

    with pytest.raises(ExternalServiceUnavailableError):
        client.get_exchange_rate("USD", "BRL")

    assert route.call_count == 2


@respx.mock
def test_persistent_server_error_is_controlled() -> None:
    route = respx.get(QUOTE_URL).mock(return_value=httpx.Response(503))
    client = AwesomeApiClient(BASE_URL)

    with pytest.raises(ExternalServiceUnavailableError):
        client.get_exchange_rate("USD", "BRL")

    assert route.call_count == 2


@respx.mock
def test_client_error_does_not_retry() -> None:
    route = respx.get(QUOTE_URL).mock(return_value=httpx.Response(404))
    client = AwesomeApiClient(BASE_URL)

    with pytest.raises(IntegrationError, match="rejected"):
        client.get_exchange_rate("USD", "BRL")

    assert route.call_count == 1


@respx.mock
def test_invalid_json_is_controlled_without_retry() -> None:
    route = respx.get(QUOTE_URL).mock(
        return_value=httpx.Response(
            200,
            content=b"not-json",
            headers={"content-type": "application/json"},
        )
    )
    client = AwesomeApiClient(BASE_URL)

    with pytest.raises(IntegrationError, match="invalid exchange rate response"):
        client.get_exchange_rate("USD", "BRL")

    assert route.call_count == 1


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"USDBRL": {"code": "USD", "codein": "BRL"}},
        quote_payload(bid="0"),
        quote_payload(bid="invalid"),
        quote_payload(timestamp="0"),
        quote_payload(timestamp="invalid"),
        {"USDBRL": {**quote_payload()["USDBRL"], "code": "EUR"}},
    ],
)
@respx.mock
def test_invalid_schema_is_controlled(payload: object) -> None:
    respx.get(QUOTE_URL).mock(return_value=httpx.Response(200, json=payload))
    client = AwesomeApiClient(BASE_URL)

    with pytest.raises(IntegrationError, match="invalid exchange rate response"):
        client.get_exchange_rate("USD", "BRL")


@dataclass
class FakeExchangeRateProvider:
    """Provedor controlado para testar autorizacao do servico."""

    quote: ExchangeQuote
    calls: list[tuple[str, str]] = field(default_factory=list)

    def get_exchange_rate(
        self, base_currency: str, quote_currency: str
    ) -> ExchangeQuote:
        """Registra o par e retorna a cotacao configurada."""
        self.calls.append((base_currency, quote_currency))
        return self.quote


def test_exchange_service_requires_authentication() -> None:
    quote = ExchangeQuote(
        base_currency="USD",
        quote_currency="BRL",
        rate=Decimal("5.7276"),
        source="AwesomeAPI",
        quoted_at=datetime.fromtimestamp(int(TIMESTAMP), tz=UTC),
    )
    provider = FakeExchangeRateProvider(quote)
    service = ExchangeService(provider)

    with pytest.raises(AuthorizationError):
        service.get_exchange_rate(ConversationState(), "USD", "BRL")

    assert provider.calls == []


def test_exchange_service_returns_confirmed_quote() -> None:
    quote = ExchangeQuote(
        base_currency="USD",
        quote_currency="BRL",
        rate=Decimal("5.7276"),
        source="AwesomeAPI",
        quoted_at=datetime.fromtimestamp(int(TIMESTAMP), tz=UTC),
    )
    provider = FakeExchangeRateProvider(quote)
    service = ExchangeService(provider)
    client = Client(
        cpf="01234567890",
        birth_date=date(1990, 5, 20),
        credit_limit=Decimal("2500.00"),
        credit_score=700,
    )

    result = service.get_exchange_rate(
        ConversationState(authenticated_client=client),
        "USD",
        "BRL",
    )

    assert result.quote is quote
    assert provider.calls == [("USD", "BRL")]


def test_unavailability_is_logged_with_the_status(
    respx_mock: respx.MockRouter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Sem esse registro, produção só mostra a frase genérica ao cliente."""
    respx_mock.get("https://api.local/json/last/USD-BRL").mock(
        return_value=httpx.Response(429)
    )
    client = AwesomeApiClient("https://api.local")

    with (
        caplog.at_level(logging.WARNING, logger="banco_agil.awesomeapi"),
        pytest.raises(ExternalServiceUnavailableError),
    ):
        client.get_exchange_rate("USD", "BRL")

    registros = [r for r in caplog.records if r.name == "banco_agil.awesomeapi"]
    assert len(registros) == 2  # uma por tentativa
    contexto = getattr(registros[-1], "audit", {})
    assert contexto["status"] == 429
    assert contexto["pair"] == "USD-BRL"
    # O status precisa estar na mensagem: formatadores simples, como o do
    # Streamlit Cloud, nao imprimem o contexto estruturado.
    assert "status=429" in registros[-1].getMessage()
    assert "pair=USD-BRL" in registros[-1].getMessage()
