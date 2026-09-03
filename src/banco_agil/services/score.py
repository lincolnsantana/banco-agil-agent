"""Calculo puro e deterministico do score de credito."""

from decimal import ROUND_HALF_UP, Decimal

from banco_agil.domain.enums import EmploymentType
from banco_agil.domain.models import CreditInterview

INCOME_WEIGHT = Decimal("30")
EMPLOYMENT_WEIGHTS = {
    EmploymentType.FORMAL: Decimal("300"),
    EmploymentType.SELF_EMPLOYED: Decimal("200"),
    EmploymentType.UNEMPLOYED: Decimal("0"),
}
DEPENDENT_WEIGHTS = {
    0: Decimal("100"),
    1: Decimal("80"),
    2: Decimal("60"),
}
THREE_OR_MORE_DEPENDENTS_WEIGHT = Decimal("30")
DEBT_WEIGHTS = {
    True: Decimal("-100"),
    False: Decimal("100"),
}
MINIMUM_SCORE = 0
MAXIMUM_SCORE = 1000


def calculate_credit_score(interview: CreditInterview) -> int:
    """Calcula, arredonda e limita o score ao intervalo de zero a mil."""
    dependent_weight = DEPENDENT_WEIGHTS.get(
        interview.dependents,
        THREE_OR_MORE_DEPENDENTS_WEIGHT,
    )
    score = (
        (interview.monthly_income / (interview.monthly_expenses + 1)) * INCOME_WEIGHT
        + EMPLOYMENT_WEIGHTS[interview.employment_type]
        + dependent_weight
        + DEBT_WEIGHTS[interview.has_active_debts]
    )
    rounded_score = int(score.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return min(max(rounded_score, MINIMUM_SCORE), MAXIMUM_SCORE)
