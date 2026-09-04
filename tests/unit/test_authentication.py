"""Testes do servico de autenticacao."""

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

import pytest

from banco_agil.agents.state import ConversationState
from banco_agil.domain.models import Client
from banco_agil.services.authentication import AuthenticationService


@dataclass
class FakeClientRepository:
    """Repositorio controlado para observar consultas sem acessar CSV."""

    client: Client | None
    searched_cpfs: list[str] = field(default_factory=list)

    def find_by_cpf(self, cpf: str) -> Client | None:
        """Registra a consulta e retorna o cliente configurado."""
        self.searched_cpfs.append(cpf)
        return self.client

    def update_credit_score(self, cpf: str, credit_score: int) -> Client:
        """Nao utilizado pelos cenarios de autenticacao."""
        raise NotImplementedError


@pytest.fixture
def trusted_client() -> Client:
    """Cria um cliente ficticio proveniente do repositorio."""
    return Client(
        cpf="01234567890",
        birth_date=date(1990, 5, 20),
        credit_limit=Decimal("2500.00"),
        credit_score=700,
    )


def test_success_authenticates_trusted_client_and_resets_attempts(
    trusted_client: Client,
) -> None:
    repository = FakeClientRepository(trusted_client)
    service = AuthenticationService(repository)
    state = ConversationState(authentication_attempts=2)

    result = service.authenticate(state, "012.345.678-90", "20/05/1990")

    assert result.authenticated
    assert result.client is trusted_client
    assert not result.should_end
    assert result.attempts == 0
    assert state.authenticated_client is trusted_client
    assert state.authentication_attempts == 0
    assert repository.searched_cpfs == ["01234567890"]


@pytest.mark.parametrize(
    ("cpf", "birth_date"),
    [
        ("123", "20/05/1990"),
        ("0123456789A", "20/05/1990"),
        ("01234567890", "1990-05-20"),
        ("01234567890", "1/05/1990"),
        ("01234567890", "01/01/2999"),
    ],
)
def test_invalid_input_fails_before_repository(
    trusted_client: Client,
    cpf: str,
    birth_date: str,
) -> None:
    repository = FakeClientRepository(trusted_client)
    service = AuthenticationService(repository)
    state = ConversationState()

    result = service.authenticate(state, cpf, birth_date)

    assert not result.authenticated
    assert result.attempts == 1
    assert not result.should_end
    assert result.client is None
    assert repository.searched_cpfs == []


def test_unknown_cpf_and_wrong_birth_date_have_same_result(
    trusted_client: Client,
) -> None:
    unknown_service = AuthenticationService(FakeClientRepository(None))
    mismatch_service = AuthenticationService(FakeClientRepository(trusted_client))

    unknown = unknown_service.authenticate(
        ConversationState(), "11111111111", "20/05/1990"
    )
    mismatch = mismatch_service.authenticate(
        ConversationState(), "01234567890", "20/05/1991"
    )

    assert unknown == mismatch
    assert unknown.model_dump() == {
        "authenticated": False,
        "attempts": 1,
        "should_end": False,
        "client": None,
    }


def test_third_consecutive_failure_signals_end(trusted_client: Client) -> None:
    repository = FakeClientRepository(trusted_client)
    service = AuthenticationService(repository)
    state = ConversationState()

    first = service.authenticate(state, "01234567890", "20/05/1991")
    second = service.authenticate(state, "01234567890", "20/05/1991")
    third = service.authenticate(state, "01234567890", "20/05/1991")

    assert not first.should_end
    assert not second.should_end
    assert third.should_end
    assert third.attempts == 3
    assert state.authentication_attempts == 3


def test_attempt_after_limit_does_not_query_repository(
    trusted_client: Client,
) -> None:
    repository = FakeClientRepository(trusted_client)
    service = AuthenticationService(repository)
    state = ConversationState(authentication_attempts=3)

    result = service.authenticate(state, "01234567890", "20/05/1990")

    assert result.should_end
    assert result.attempts == 3
    assert repository.searched_cpfs == []


def test_repository_identity_must_match_requested_cpf(
    trusted_client: Client,
) -> None:
    repository = FakeClientRepository(trusted_client)
    service = AuthenticationService(repository)
    state = ConversationState()

    result = service.authenticate(state, "11111111111", "20/05/1990")

    assert not result.authenticated
    assert result.client is None
    assert state.authenticated_client is None
    assert state.authentication_attempts == 1
