"""Protocolos de persistencia consumidos pelos servicos."""

from typing import Protocol

from banco_agil.domain.models import Client


class ClientRepository(Protocol):
    """Contrato para consulta e atualizacao de clientes."""

    def find_by_cpf(self, cpf: str) -> Client | None:
        """Retorna o cliente do CPF informado quando existente."""
        ...

    def update_credit_score(self, cpf: str, credit_score: int) -> Client:
        """Atualiza o score e retorna o cliente persistido."""
        ...
