"""Regras de autenticacao sem revelar qual credencial falhou."""

import re
from datetime import date

from banco_agil.agents.state import ConversationState
from banco_agil.domain.models import AuthenticationResult
from banco_agil.repositories.protocols import ClientRepository

MAX_AUTHENTICATION_ATTEMPTS = 3


class AuthenticationService:
    """Autentica clientes usando somente a identidade confiavel do repositorio."""

    def __init__(self, client_repository: ClientRepository) -> None:
        """Recebe o repositorio sem depender de sua implementacao concreta."""
        self._client_repository = client_repository

    def authenticate(
        self,
        state: ConversationState,
        cpf: str,
        birth_date: str,
    ) -> AuthenticationResult:
        """Valida credenciais e atualiza as tentativas da sessao.

        O resultado de falha e identico para CPF inexistente, nascimento incorreto
        ou formato invalido.
        """
        if state.authentication_attempts >= MAX_AUTHENTICATION_ATTEMPTS:
            return self._failure_result(state)

        try:
            normalized_cpf = _normalize_cpf(cpf)
            parsed_birth_date = _parse_birth_date(birth_date)
        except ValueError:
            return self._register_failure(state)

        client = self._client_repository.find_by_cpf(normalized_cpf)
        if client is None or client.birth_date != parsed_birth_date:
            return self._register_failure(state)

        state.authenticated_client = client
        state.authentication_attempts = 0
        state.pending_cpf = None
        state.pending_birth_date = None
        return AuthenticationResult(
            authenticated=True,
            attempts=0,
            should_end=False,
            client=client,
        )

    @staticmethod
    def _register_failure(state: ConversationState) -> AuthenticationResult:
        state.authentication_attempts += 1
        return AuthenticationService._failure_result(state)

    @staticmethod
    def _failure_result(state: ConversationState) -> AuthenticationResult:
        return AuthenticationResult(
            authenticated=False,
            attempts=state.authentication_attempts,
            should_end=(state.authentication_attempts >= MAX_AUTHENTICATION_ATTEMPTS),
        )


def _normalize_cpf(cpf: str) -> str:
    normalized = re.sub(r"[.\-\s]", "", cpf)
    if not re.fullmatch(r"\d{11}", normalized):
        raise ValueError("invalid CPF")
    return normalized


def _parse_birth_date(value: str) -> date:
    normalized = value.strip()
    parsed = date.fromisoformat(normalized)
    if parsed.isoformat() != normalized or parsed > date.today():
        raise ValueError("invalid birth date")
    return parsed
