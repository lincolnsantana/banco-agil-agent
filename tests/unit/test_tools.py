"""Testes dos wrappers tipados das tools bancarias."""

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from langchain_core.tools import BaseTool

from banco_agil.agents.state import ConversationState
from banco_agil.domain.enums import CreditRequestStatus, EmploymentType, EndReason
from banco_agil.domain.exceptions import AuthorizationError
from banco_agil.domain.models import (
    AuthenticationResult,
    Client,
    CreditInterview,
    CreditLimitResult,
    ExchangeQuote,
    ExchangeRateResult,
    LimitIncreaseResult,
    ScoreUpdateResult,
)
from banco_agil.tools.banking import (
    authenticate_client,
    end_service,
    get_credit_limit,
    get_exchange_rate,
    request_limit_increase,
    update_credit_score,
)


@dataclass
class FakeAuthenticationService:
    """Registra credenciais e devolve um resultado configurado."""

    result: AuthenticationResult
    calls: list[tuple[ConversationState, str, str]] = field(default_factory=list)

    def authenticate(
        self,
        state: ConversationState,
        cpf: str,
        birth_date: str,
    ) -> AuthenticationResult:
        """Registra a delegacao da tool."""
        self.calls.append((state, cpf, birth_date))
        return self.result


@dataclass
class FakeCreditService:
    """Registra consultas e solicitacoes de credito."""

    limit_calls: list[ConversationState] = field(default_factory=list)
    request_calls: list[tuple[ConversationState, Decimal]] = field(default_factory=list)

    def get_credit_limit(self, state: ConversationState) -> CreditLimitResult:
        """Retorna um limite ficticio."""
        self.limit_calls.append(state)
        return CreditLimitResult(current_limit=Decimal("2500.00"))

    def request_limit_increase(
        self,
        state: ConversationState,
        new_limit: Decimal,
    ) -> LimitIncreaseResult:
        """Retorna uma aprovacao ficticia."""
        self.request_calls.append((state, new_limit))
        return LimitIncreaseResult(
            current_limit=Decimal("2500.00"),
            requested_limit=new_limit,
            status=CreditRequestStatus.APPROVED,
        )


@dataclass
class FakeCreditInterviewService:
    """Registra entrevista validada recebida pela tool."""

    calls: list[tuple[ConversationState, CreditInterview]] = field(default_factory=list)

    def update_credit_score(
        self,
        state: ConversationState,
        interview: CreditInterview,
    ) -> ScoreUpdateResult:
        """Retorna uma atualizacao ficticia."""
        self.calls.append((state, interview))
        return ScoreUpdateResult(previous_score=700, new_score=760)


@dataclass
class FakeExchangeService:
    """Registra o par de moedas recebido pela tool."""

    quote: ExchangeQuote
    calls: list[tuple[ConversationState, str, str]] = field(default_factory=list)

    def get_exchange_rate(
        self,
        state: ConversationState,
        base_currency: str,
        quote_currency: str,
    ) -> ExchangeRateResult:
        """Retorna uma cotacao ficticia."""
        self.calls.append((state, base_currency, quote_currency))
        return ExchangeRateResult(quote=self.quote)


@pytest.fixture
def authenticated_state() -> ConversationState:
    """Cria um estado com cliente ficticio confiavel."""
    client = Client(
        cpf="01234567890",
        birth_date=date(1990, 5, 20),
        credit_limit=Decimal("2500.00"),
        credit_score=700,
    )
    return ConversationState(
        authenticated_client=client,
        requested_limit=Decimal("5000.00"),
    )


@pytest.fixture
def interview() -> CreditInterview:
    """Cria uma entrevista completa e validada."""
    return CreditInterview(
        monthly_income=Decimal("5000.00"),
        employment_type=EmploymentType.FORMAL,
        monthly_expenses=Decimal("2000.00"),
        dependents=1,
        has_active_debts=False,
    )


def test_llm_schemas_hide_state_and_dependencies() -> None:
    schemas = {
        authenticate_client.name: {"cpf", "birth_date"},
        get_credit_limit.name: set(),
        request_limit_increase.name: {"new_limit"},
        update_credit_score.name: {"interview"},
        get_exchange_rate.name: {"base_currency", "quote_currency"},
        end_service.name: {"reason"},
    }

    for banking_tool in (
        authenticate_client,
        get_credit_limit,
        request_limit_increase,
        update_credit_score,
        get_exchange_rate,
        end_service,
    ):
        schema = banking_tool.tool_call_schema.model_json_schema()
        assert set(schema["properties"]) == schemas[banking_tool.name]


def test_tool_descriptions_are_single_short_sentences() -> None:
    expected_descriptions = {
        "authenticate_client": "Valida CPF e nascimento informados",
        "get_credit_limit": "Consulta o limite do cliente autenticado",
        "request_limit_increase": "Registra e avalia o limite solicitado",
        "update_credit_score": "Calcula e atualiza o score após entrevista",
        "get_exchange_rate": "Consulta a cotação atual de um par de moedas",
        "end_service": "Encerra o atendimento atual",
    }

    for banking_tool in (
        authenticate_client,
        get_credit_limit,
        request_limit_increase,
        update_credit_score,
        get_exchange_rate,
        end_service,
    ):
        assert banking_tool.description == expected_descriptions[banking_tool.name]


def test_authentication_tool_delegates_credentials(
    authenticated_state: ConversationState,
) -> None:
    result = AuthenticationResult(
        authenticated=True,
        attempts=0,
        should_end=False,
        client=authenticated_state.authenticated_client,
    )
    service = FakeAuthenticationService(result)
    state = ConversationState()

    returned = authenticate_client.invoke(
        {
            "cpf": "01234567890",
            "birth_date": "1990-05-20",
            "state": state,
            "service": service,
        }
    )

    assert returned is result
    assert service.calls == [(state, "01234567890", "1990-05-20")]


def test_credit_tools_delegate_without_receiving_cpf(
    authenticated_state: ConversationState,
) -> None:
    service = FakeCreditService()

    limit_result = get_credit_limit.invoke(
        {"state": authenticated_state, "service": service}
    )
    request_result = request_limit_increase.invoke(
        {
            "new_limit": Decimal("4000.00"),
            "state": authenticated_state,
            "service": service,
        }
    )

    assert limit_result.current_limit == Decimal("2500.00")
    assert request_result.status is CreditRequestStatus.APPROVED
    assert service.limit_calls == [authenticated_state]
    assert service.request_calls == [(authenticated_state, Decimal("4000.00"))]


def test_score_tool_delegates_validated_interview(
    authenticated_state: ConversationState,
    interview: CreditInterview,
) -> None:
    service = FakeCreditInterviewService()

    result = update_credit_score.invoke(
        {
            "interview": interview,
            "state": authenticated_state,
            "service": service,
        }
    )

    assert result == ScoreUpdateResult(previous_score=700, new_score=760)
    assert service.calls == [(authenticated_state, interview)]


def test_exchange_tool_delegates_currency_pair(
    authenticated_state: ConversationState,
) -> None:
    quote = ExchangeQuote(
        base_currency="USD",
        quote_currency="BRL",
        rate=Decimal("5.25"),
        source="AwesomeAPI",
        quoted_at=datetime(2026, 9, 3, 12, 0, tzinfo=UTC),
    )
    service = FakeExchangeService(quote)

    result = get_exchange_rate.invoke(
        {
            "base_currency": "USD",
            "quote_currency": "BRL",
            "state": authenticated_state,
            "service": service,
        }
    )

    assert result.quote is quote
    assert service.calls == [(authenticated_state, "USD", "BRL")]


@pytest.mark.parametrize(
    ("banking_tool", "arguments"),
    [
        (get_credit_limit, {"service": FakeCreditService()}),
        (
            request_limit_increase,
            {"new_limit": Decimal("4000.00"), "service": FakeCreditService()},
        ),
        (
            update_credit_score,
            {
                "interview": CreditInterview(
                    monthly_income=Decimal("5000.00"),
                    employment_type=EmploymentType.FORMAL,
                    monthly_expenses=Decimal("2000.00"),
                    dependents=1,
                    has_active_debts=False,
                ),
                "service": FakeCreditInterviewService(),
            },
        ),
        (
            get_exchange_rate,
            {
                "base_currency": "USD",
                "quote_currency": "BRL",
                "service": FakeExchangeService(
                    ExchangeQuote(
                        base_currency="USD",
                        quote_currency="BRL",
                        rate=Decimal("5.25"),
                        source="AwesomeAPI",
                        quoted_at=datetime(2026, 9, 3, 12, 0, tzinfo=UTC),
                    )
                ),
            },
        ),
    ],
)
def test_each_protected_tool_checks_authorization(
    banking_tool: BaseTool,
    arguments: dict[str, object],
) -> None:
    arguments["state"] = ConversationState()

    with pytest.raises(AuthorizationError):
        banking_tool.invoke(arguments)


def test_end_service_returns_controlled_result() -> None:
    result = end_service.invoke({"reason": EndReason.USER_REQUEST})

    assert result.ended
    assert result.reason is EndReason.USER_REQUEST
