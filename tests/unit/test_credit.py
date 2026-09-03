"""Testes das regras deterministicas de credito."""

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from banco_agil.agents.state import ConversationState
from banco_agil.domain.enums import CreditRequestStatus
from banco_agil.domain.exceptions import AuthorizationError, DomainError
from banco_agil.domain.models import Client, CreditRequest
from banco_agil.services.credit import CreditService

FIXED_TIME = datetime(2026, 9, 2, 15, 0, tzinfo=UTC)


@dataclass
class FakeScoreLimitRepository:
    """Retorna um limite fixo e registra o score consultado."""

    maximum_limit: Decimal
    queried_scores: list[int] = field(default_factory=list)

    def find_max_limit(self, score: int) -> Decimal:
        """Registra e responde a consulta da faixa."""
        self.queried_scores.append(score)
        return self.maximum_limit


@dataclass
class FakeCreditRequestRepository:
    """Simula a mesma linha persistida durante criacao e finalizacao."""

    requests: list[CreditRequest] = field(default_factory=list)
    created_statuses: list[CreditRequestStatus] = field(default_factory=list)
    statuses_received: list[CreditRequestStatus] = field(default_factory=list)

    def create(self, request: CreditRequest) -> CreditRequest:
        """Registra uma solicitacao pendente."""
        self.created_statuses.append(request.status)
        self.requests.append(request)
        return request

    def finalize(
        self,
        request: CreditRequest,
        status: CreditRequestStatus,
    ) -> CreditRequest:
        """Substitui a solicitacao existente pela versao final."""
        self.statuses_received.append(status)
        finalized = request.model_copy(update={"status": status})
        self.requests[0] = finalized
        return finalized


@pytest.fixture
def authenticated_state() -> ConversationState:
    """Cria uma sessao com cliente ficticio confiavel."""
    client = Client(
        cpf="01234567890",
        birth_date=date(1990, 5, 20),
        credit_limit=Decimal("2500.00"),
        credit_score=700,
    )
    return ConversationState(authenticated_client=client)


def build_service(
    maximum_limit: Decimal = Decimal("5000.00"),
) -> tuple[CreditService, FakeScoreLimitRepository, FakeCreditRequestRepository]:
    """Monta o servico com dependencias observaveis."""
    score_repository = FakeScoreLimitRepository(maximum_limit)
    request_repository = FakeCreditRequestRepository()
    service = CreditService(
        score_limit_repository=score_repository,
        credit_request_repository=request_repository,
        clock=lambda: FIXED_TIME,
    )
    return service, score_repository, request_repository


def test_credit_limit_requires_authenticated_session() -> None:
    service, _, _ = build_service()

    with pytest.raises(AuthorizationError):
        service.get_credit_limit(ConversationState())


def test_get_credit_limit_uses_trusted_client(
    authenticated_state: ConversationState,
) -> None:
    service, _, _ = build_service()

    result = service.get_credit_limit(authenticated_state)

    assert result.current_limit == Decimal("2500.00")


def test_limit_increase_requires_authenticated_session() -> None:
    service, score_repository, request_repository = build_service()

    with pytest.raises(AuthorizationError):
        service.request_limit_increase(ConversationState(), Decimal("3000.00"))

    assert score_repository.queried_scores == []
    assert request_repository.requests == []


@pytest.mark.parametrize(
    "new_limit",
    [
        Decimal("-1.00"),
        Decimal("0"),
        Decimal("2000.00"),
        Decimal("2500.00"),
        Decimal("2500.004"),
        Decimal("NaN"),
        Decimal("Infinity"),
    ],
)
def test_invalid_new_limit_is_rejected_without_persistence(
    authenticated_state: ConversationState,
    new_limit: Decimal,
) -> None:
    service, score_repository, request_repository = build_service()

    with pytest.raises(DomainError, match="greater than current limit"):
        service.request_limit_increase(authenticated_state, new_limit)

    assert score_repository.queried_scores == []
    assert request_repository.requests == []


def test_request_is_approved_at_score_limit_boundary(
    authenticated_state: ConversationState,
) -> None:
    service, score_repository, request_repository = build_service(Decimal("5000.00"))
    original_client = authenticated_state.authenticated_client

    result = service.request_limit_increase(
        authenticated_state,
        Decimal("5000.00"),
    )

    assert result.status is CreditRequestStatus.APPROVED
    assert not result.offer_interview
    assert request_repository.statuses_received == [CreditRequestStatus.APPROVED]
    assert request_repository.requests[0].status is CreditRequestStatus.APPROVED
    assert request_repository.requests[0].requested_at == FIXED_TIME
    assert score_repository.queried_scores == [700]
    assert authenticated_state.authenticated_client is original_client
    assert authenticated_state.authenticated_client.credit_limit == Decimal("2500.00")


def test_decision_uses_same_cent_value_that_is_persisted(
    authenticated_state: ConversationState,
) -> None:
    service, _, request_repository = build_service(Decimal("5000.00"))

    result = service.request_limit_increase(
        authenticated_state,
        Decimal("5000.004"),
    )

    assert result.requested_limit == Decimal("5000.00")
    assert result.status is CreditRequestStatus.APPROVED
    assert request_repository.requests[0].requested_limit == Decimal("5000.00")


def test_rejected_request_offers_credit_interview(
    authenticated_state: ConversationState,
) -> None:
    service, _, request_repository = build_service(Decimal("4999.99"))

    result = service.request_limit_increase(
        authenticated_state,
        Decimal("5000.00"),
    )

    assert result.status is CreditRequestStatus.REJECTED
    assert result.offer_interview
    assert request_repository.statuses_received == [CreditRequestStatus.REJECTED]
    assert len(request_repository.requests) == 1
    assert authenticated_state.requested_limit == Decimal("5000.00")


def test_request_is_created_as_pending_before_decision(
    authenticated_state: ConversationState,
) -> None:
    service, _, request_repository = build_service()

    service.request_limit_increase(authenticated_state, Decimal("3000.00"))

    assert request_repository.created_statuses == [CreditRequestStatus.PENDING]
