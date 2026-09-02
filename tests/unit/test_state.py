"""Testes do estado da conversa."""

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from banco_agil.agents.state import ConversationState, CreditInterviewDraft
from banco_agil.domain.enums import Agent, EmploymentType, EndReason, Intent
from banco_agil.domain.models import Client


def test_state_starts_in_triage_without_authentication() -> None:
    state = ConversationState()

    assert state.active_agent is Agent.TRIAGE
    assert state.intent is Intent.UNKNOWN
    assert state.authentication_attempts == 0
    assert not state.authenticated
    assert not state.ended


def test_state_representation_and_log_context_omit_sensitive_data() -> None:
    client = Client(
        cpf="01234567890",
        birth_date=date(1990, 5, 20),
        credit_limit=Decimal("2500.00"),
        credit_score=700,
    )
    state = ConversationState(
        active_agent=Agent.CREDIT_INTERVIEW,
        authenticated_client=client,
        requested_limit=Decimal("4000.00"),
        interview_draft=CreditInterviewDraft(
            consent_given=True,
            monthly_income=Decimal("5000.00"),
            employment_type=EmploymentType.FORMAL,
        ),
    )

    representation = repr(state)
    log_context = state.to_log_context()

    assert "01234567890" not in representation
    assert "5000.00" not in representation
    assert "01234567890" not in str(log_context)
    assert "5000.00" not in str(log_context)
    assert log_context["authenticated"] is True


def test_partial_interview_requires_consent() -> None:
    with pytest.raises(ValidationError):
        CreditInterviewDraft(
            consent_given=False,
            monthly_income=Decimal("5000.00"),
        )


def test_ended_state_requires_reason() -> None:
    with pytest.raises(ValidationError):
        ConversationState(ended=True)

    state = ConversationState(ended=True, end_reason=EndReason.USER_REQUEST)

    assert state.ended
    assert state.end_reason is EndReason.USER_REQUEST


@pytest.mark.parametrize("attempts", [-1, 4])
def test_state_rejects_invalid_authentication_attempts(attempts: int) -> None:
    with pytest.raises(ValidationError):
        ConversationState(authentication_attempts=attempts)
