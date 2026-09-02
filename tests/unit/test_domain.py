"""Testes dos contratos de dominio."""

import ast
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from banco_agil.domain.enums import CreditRequestStatus, EmploymentType
from banco_agil.domain.models import (
    AuthenticationResult,
    Client,
    CreditInterview,
    CreditRequest,
    ExchangeQuote,
)


def test_client_accepts_valid_banking_data() -> None:
    client = Client(
        cpf="01234567890",
        birth_date=date(1990, 5, 20),
        credit_limit=Decimal("2500.00"),
        credit_score=700,
    )

    assert client.cpf == "01234567890"
    assert client.credit_limit == Decimal("2500.00")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cpf", "123"),
        ("birth_date", date.today() + timedelta(days=1)),
        ("credit_limit", Decimal("NaN")),
        ("credit_limit", Decimal("-0.01")),
        ("credit_score", 1001),
    ],
)
def test_client_rejects_invalid_data(field: str, value: object) -> None:
    data: dict[str, object] = {
        "cpf": "01234567890",
        "birth_date": date(1990, 5, 20),
        "credit_limit": Decimal("2500.00"),
        "credit_score": 700,
    }
    data[field] = value

    with pytest.raises(ValidationError):
        Client.model_validate(data)


def test_credit_request_normalizes_timestamp_to_utc() -> None:
    local_timezone = timezone(timedelta(hours=-3))
    request = CreditRequest(
        client_cpf="01234567890",
        requested_at=datetime(2026, 9, 2, 9, 0, tzinfo=local_timezone),
        current_limit=Decimal("1000.00"),
        requested_limit=Decimal("2000.00"),
        status=CreditRequestStatus.PENDING,
    )

    assert request.requested_at == datetime(2026, 9, 2, 12, 0, tzinfo=UTC)


def test_models_reject_naive_timestamps_and_invalid_money() -> None:
    with pytest.raises(ValidationError):
        CreditRequest(
            client_cpf="01234567890",
            requested_at=datetime(2026, 9, 2, 12, 0),
            current_limit=Decimal("1000.00"),
            requested_limit=Decimal("0"),
            status=CreditRequestStatus.PENDING,
        )


def test_interview_and_quote_validate_their_ranges() -> None:
    interview = CreditInterview(
        monthly_income=Decimal("5000.00"),
        employment_type=EmploymentType.FORMAL,
        monthly_expenses=Decimal("2000.00"),
        dependents=2,
        has_active_debts=False,
    )
    quote = ExchangeQuote(
        base_currency="usd",
        quote_currency="brl",
        rate=Decimal("5.25"),
        source="AwesomeAPI",
        quoted_at=datetime(2026, 9, 2, 12, 0, tzinfo=UTC),
    )

    assert interview.monthly_income == Decimal("5000.00")
    assert quote.base_currency == "USD"
    assert quote.quote_currency == "BRL"

    with pytest.raises(ValidationError):
        ExchangeQuote(
            base_currency="USD",
            quote_currency="BRL",
            rate=Decimal("0"),
            source="AwesomeAPI",
            quoted_at=datetime(2026, 9, 2, 12, 0, tzinfo=UTC),
        )


def test_authentication_result_hides_client_from_representation() -> None:
    client = Client(
        cpf="01234567890",
        birth_date=date(1990, 5, 20),
        credit_limit=Decimal("2500.00"),
        credit_score=700,
    )
    result = AuthenticationResult(
        authenticated=True,
        attempts=0,
        should_end=False,
        client=client,
    )

    assert "01234567890" not in repr(result)


def test_domain_does_not_import_application_frameworks() -> None:
    domain_path = Path(__file__).parents[2] / "src" / "banco_agil" / "domain"
    forbidden_modules = {"streamlit", "langgraph", "httpx", "groq"}
    imported_modules: set[str] = set()

    for source_path in domain_path.glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(
                    alias.name.split(".")[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module.split(".")[0])

    assert imported_modules.isdisjoint(forbidden_modules)
