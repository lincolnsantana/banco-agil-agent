"""Testes do fluxo incremental da entrevista de credito."""

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

import pytest

from banco_agil.agents.state import ConversationState, CreditInterviewDraft
from banco_agil.domain.enums import Agent
from banco_agil.domain.exceptions import AuthorizationError, DomainError
from banco_agil.domain.models import Client
from banco_agil.services.credit_interview import (
    CreditInterviewService,
    InterviewField,
)


@dataclass
class FakeClientRepository:
    """Atualiza um cliente ficticio e registra a identidade usada."""

    client: Client
    updates: list[tuple[str, int]] = field(default_factory=list)

    def find_by_cpf(self, cpf: str) -> Client | None:
        """Retorna o cliente configurado quando o CPF coincide."""
        return self.client if cpf == self.client.cpf else None

    def update_credit_score(self, cpf: str, credit_score: int) -> Client:
        """Registra e devolve o cliente com o novo score."""
        self.updates.append((cpf, credit_score))
        self.client = Client(
            cpf=self.client.cpf,
            birth_date=self.client.birth_date,
            credit_limit=self.client.credit_limit,
            credit_score=credit_score,
        )
        return self.client


@pytest.fixture
def interview_context() -> tuple[
    CreditInterviewService,
    FakeClientRepository,
    ConversationState,
]:
    """Monta uma sessao apta a iniciar entrevista apos rejeicao."""
    client = Client(
        cpf="01234567890",
        birth_date=date(1990, 5, 20),
        credit_limit=Decimal("2500.00"),
        credit_score=700,
    )
    repository = FakeClientRepository(client)
    state = ConversationState(
        authenticated_client=client,
        requested_limit=Decimal("5000.00"),
    )
    return CreditInterviewService(repository), repository, state


def test_interview_requires_authenticated_client(
    interview_context: tuple[
        CreditInterviewService,
        FakeClientRepository,
        ConversationState,
    ],
) -> None:
    service, _, _ = interview_context

    with pytest.raises(AuthorizationError):
        service.start(ConversationState(), consent=True)


def test_interview_requires_rejected_limit_to_reanalyze(
    interview_context: tuple[
        CreditInterviewService,
        FakeClientRepository,
        ConversationState,
    ],
) -> None:
    service, _, state = interview_context
    state.requested_limit = None

    with pytest.raises(DomainError, match="requested limit"):
        service.start(state, consent=True)


def test_declined_consent_does_not_store_or_persist_answers(
    interview_context: tuple[
        CreditInterviewService,
        FakeClientRepository,
        ConversationState,
    ],
) -> None:
    service, repository, state = interview_context

    progress = service.start(state, consent=False)

    assert progress.consent_declined
    assert progress.next_field is None
    assert state.interview_draft == CreditInterviewDraft()
    assert repository.updates == []


def test_answer_cannot_be_submitted_without_consent(
    interview_context: tuple[
        CreditInterviewService,
        FakeClientRepository,
        ConversationState,
    ],
) -> None:
    service, _, state = interview_context

    with pytest.raises(DomainError, match="consent"):
        service.submit_answer(state, "5000")


@pytest.mark.parametrize(
    ("valid_answers", "invalid_answer"),
    [
        ([], "-1"),
        (["5000"], "informal"),
        (["5000", "formal"], "-1"),
        (["5000", "formal", "2000"], "2.5"),
        (["5000", "formal", "2000", "2"], "talvez"),
    ],
)
def test_invalid_answer_does_not_advance_or_persist(
    interview_context: tuple[
        CreditInterviewService,
        FakeClientRepository,
        ConversationState,
    ],
    valid_answers: list[str],
    invalid_answer: str,
) -> None:
    service, repository, state = interview_context
    service.start(state, consent=True)
    for answer in valid_answers:
        service.submit_answer(state, answer)
    previous_draft = state.interview_draft.model_copy(deep=True)

    with pytest.raises(DomainError, match="invalid interview answer"):
        service.submit_answer(state, invalid_answer)

    assert state.interview_draft == previous_draft
    assert repository.updates == []


def test_complete_interview_updates_score_and_returns_to_credit(
    interview_context: tuple[
        CreditInterviewService,
        FakeClientRepository,
        ConversationState,
    ],
) -> None:
    service, repository, state = interview_context
    progress = service.start(state, consent=True)

    assert progress.next_field is InterviewField.MONTHLY_INCOME
    expected_fields = [
        InterviewField.EMPLOYMENT_TYPE,
        InterviewField.MONTHLY_EXPENSES,
        InterviewField.DEPENDENTS,
        InterviewField.ACTIVE_DEBTS,
    ]
    for answer, expected_field in zip(
        ["5000", "formal", "2000", "3"],
        expected_fields,
        strict=True,
    ):
        progress = service.submit_answer(state, answer)
        assert progress.next_field is expected_field
        assert repository.updates == []

    progress = service.submit_answer(state, "não")

    assert progress.completed
    assert progress.score_update is not None
    assert progress.score_update.previous_score == 700
    assert progress.score_update.new_score == 505
    assert repository.updates == [("01234567890", 505)]
    assert state.authenticated_client is repository.client
    assert state.active_agent is Agent.CREDIT
    assert state.credit_reanalysis_pending
    assert state.requested_limit == Decimal("5000.00")
    assert state.interview_draft == CreditInterviewDraft()
