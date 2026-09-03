"""Caso de uso autenticado para consulta de cambio."""

from typing import Protocol

from banco_agil.agents.state import ConversationState
from banco_agil.domain.exceptions import AuthorizationError
from banco_agil.domain.models import ExchangeQuote, ExchangeRateResult


class ExchangeRateProvider(Protocol):
    """Contrato para uma fonte externa de cotacoes."""

    def get_exchange_rate(
        self,
        base_currency: str,
        quote_currency: str,
    ) -> ExchangeQuote:
        """Retorna uma cotacao atual confirmada pela fonte."""
        ...


class ExchangeService:
    """Protege e delega consultas de cambio ao provedor configurado."""

    def __init__(self, provider: ExchangeRateProvider) -> None:
        """Recebe a integracao por protocolo."""
        self._provider = provider

    def get_exchange_rate(
        self,
        state: ConversationState,
        base_currency: str,
        quote_currency: str,
    ) -> ExchangeRateResult:
        """Consulta cambio somente para uma sessao autenticada.

        Raises:
            AuthorizationError: Se nao houver cliente autenticado.
        """
        if state.authenticated_client is None:
            raise AuthorizationError("authenticated client is required")
        quote = self._provider.get_exchange_rate(base_currency, quote_currency)
        return ExchangeRateResult(quote=quote)
