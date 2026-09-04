"""Protocolos de persistencia consumidos pelos servicos."""

from decimal import Decimal
from typing import Protocol

from banco_agil.domain.enums import CreditRequestStatus
from banco_agil.domain.models import Client, CreditRequest


class ClientRepository(Protocol):
    """Contrato para consulta e atualizacao de clientes."""

    def find_by_cpf(self, cpf: str) -> Client | None:
        """Retorna o cliente do CPF informado quando existente."""
        ...

    def update_credit_score(self, cpf: str, credit_score: int) -> Client:
        """Atualiza o score e retorna o cliente persistido."""
        ...

    def update_credit_limit(self, cpf: str, credit_limit: Decimal) -> Client:
        """Atualiza o limite e retorna o cliente persistido."""
        ...


class ScoreLimitRepository(Protocol):
    """Contrato para consulta do limite permitido por score."""

    def find_max_limit(self, score: int) -> Decimal:
        """Retorna o limite maximo da faixa que contem o score."""
        ...


class CreditRequestRepository(Protocol):
    """Contrato para persistencia de solicitacoes de credito."""

    def create(self, request: CreditRequest) -> CreditRequest:
        """Persiste e retorna uma nova solicitacao pendente."""
        ...

    def finalize(
        self,
        request: CreditRequest,
        status: CreditRequestStatus,
    ) -> CreditRequest:
        """Finaliza a mesma solicitacao como aprovada ou rejeitada."""
        ...
