"""Validação antecipada de CPF e autenticação completa do cliente."""

import re
from datetime import date, datetime

from banco_agil.agents.state import ConversationState
from banco_agil.domain.models import AuthenticationResult, CpfValidationResult
from banco_agil.repositories.protocols import ClientRepository

MAX_AUTHENTICATION_ATTEMPTS = 3


class AuthenticationService:
    """Autentica clientes usando somente a identidade confiavel do repositorio."""

    def __init__(self, client_repository: ClientRepository) -> None:
        """Recebe o repositorio sem depender de sua implementacao concreta."""
        self._client_repository = client_repository

    def validate_cpf(
        self,
        state: ConversationState,
        cpf: str,
    ) -> CpfValidationResult:
        """Confirma a existência do CPF sem autenticar ou retornar o cliente."""
        if state.authentication_attempts >= MAX_AUTHENTICATION_ATTEMPTS:
            return self._cpf_failure_result(state)

        try:
            normalized_cpf = _normalize_cpf(cpf)
        except ValueError:
            return self._register_cpf_failure(state)

        client = self._client_repository.find_by_cpf(normalized_cpf)
        if client is None or client.cpf != normalized_cpf:
            return self._register_cpf_failure(state)

        state.pending_cpf = normalized_cpf
        return CpfValidationResult(
            valid=True,
            attempts=state.authentication_attempts,
            should_end=False,
        )

    def authenticate(
        self,
        state: ConversationState,
        cpf: str,
        birth_date: str,
    ) -> AuthenticationResult:
        """Valida credenciais e atualiza as tentativas da sessao.

        O resultado não expõe dados cadastrais e só autentica quando CPF e
        nascimento correspondem ao mesmo cliente.
        """
        if state.authentication_attempts >= MAX_AUTHENTICATION_ATTEMPTS:
            return self._failure_result(state)

        try:
            normalized_cpf = _normalize_cpf(cpf)
            parsed_birth_date = _parse_birth_date(birth_date)
        except ValueError:
            return self._register_failure(state)

        client = self._client_repository.find_by_cpf(normalized_cpf)
        if (
            client is None
            or client.cpf != normalized_cpf
            or client.birth_date != parsed_birth_date
        ):
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

    @staticmethod
    def _register_cpf_failure(state: ConversationState) -> CpfValidationResult:
        state.pending_cpf = None
        state.authentication_attempts += 1
        return AuthenticationService._cpf_failure_result(state)

    @staticmethod
    def _cpf_failure_result(state: ConversationState) -> CpfValidationResult:
        return CpfValidationResult(
            valid=False,
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
    if not re.fullmatch(r"\d{2}/\d{2}/\d{4}", normalized):
        raise ValueError("invalid birth date")
    parsed = datetime.strptime(normalized, "%d/%m/%Y").date()
    if parsed > date.today():
        raise ValueError("invalid birth date")
    return parsed
