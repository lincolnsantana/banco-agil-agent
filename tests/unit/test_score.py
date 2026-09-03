"""Testes do calculo puro de score de credito."""

from decimal import Decimal

from banco_agil.domain.enums import EmploymentType
from banco_agil.domain.models import CreditInterview
from banco_agil.services.score import calculate_credit_score


def build_interview(
    monthly_income: str,
    employment_type: EmploymentType,
    monthly_expenses: str,
    dependents: int,
    has_active_debts: bool,
) -> CreditInterview:
    """Cria uma entrevista validada para os cenarios de score."""
    return CreditInterview(
        monthly_income=Decimal(monthly_income),
        employment_type=employment_type,
        monthly_expenses=Decimal(monthly_expenses),
        dependents=dependents,
        has_active_debts=has_active_debts,
    )


def test_score_uses_all_required_weights() -> None:
    interview = build_interview(
        "5000.00",
        EmploymentType.FORMAL,
        "2000.00",
        2,
        False,
    )

    assert calculate_credit_score(interview) == 535


def test_three_or_more_dependents_share_same_weight() -> None:
    three_dependents = build_interview(
        "3000.00",
        EmploymentType.SELF_EMPLOYED,
        "1000.00",
        3,
        False,
    )
    six_dependents = three_dependents.model_copy(update={"dependents": 6})

    assert calculate_credit_score(three_dependents) == calculate_credit_score(
        six_dependents
    )


def test_score_is_limited_to_zero_and_one_thousand() -> None:
    minimum = build_interview(
        "0",
        EmploymentType.UNEMPLOYED,
        "0",
        5,
        True,
    )
    maximum = build_interview(
        "100000",
        EmploymentType.FORMAL,
        "0",
        0,
        False,
    )

    assert calculate_credit_score(minimum) == 0
    assert calculate_credit_score(maximum) == 1000


def test_score_rounds_half_up_deterministically() -> None:
    interview = build_interview(
        "1",
        EmploymentType.FORMAL,
        "59",
        0,
        False,
    )

    assert calculate_credit_score(interview) == 501
    assert calculate_credit_score(interview) == 501
