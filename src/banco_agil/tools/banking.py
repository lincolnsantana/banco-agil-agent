"""Wrappers pequenos e tipados sobre os servicos bancarios."""

from decimal import Decimal
from typing import Annotated

from langchain_core.tools import InjectedToolArg, tool
from pydantic import SkipValidation

from banco_agil.agents.state import ConversationState
from banco_agil.domain.enums import EndReason
from banco_agil.domain.exceptions import AuthorizationError
from banco_agil.domain.models import (
    AuthenticationResult,
    CpfValidationResult,
    CreditInterview,
    CreditLimitResult,
    EndServiceResult,
    ExchangeRateResult,
    LimitIncreaseResult,
    ScoreUpdateResult,
)
from banco_agil.services.authentication import AuthenticationService
from banco_agil.services.credit import CreditService
from banco_agil.services.credit_interview import CreditInterviewService
from banco_agil.services.exchange import ExchangeService

InjectedState = Annotated[ConversationState, InjectedToolArg]
InjectedAuthenticationService = Annotated[
    SkipValidation[AuthenticationService], InjectedToolArg
]
InjectedCreditService = Annotated[SkipValidation[CreditService], InjectedToolArg]
InjectedInterviewService = Annotated[
    SkipValidation[CreditInterviewService], InjectedToolArg
]
InjectedExchangeService = Annotated[SkipValidation[ExchangeService], InjectedToolArg]


@tool(
    "validate_client_cpf",
    description="Confirma se o CPF informado existe no cadastro",
)
def validate_client_cpf(
    cpf: str,
    state: InjectedState,
    service: InjectedAuthenticationService,
) -> CpfValidationResult:
    """Delega a verificação cadastral do CPF sem autenticar o cliente."""
    return service.validate_cpf(state, cpf)


@tool(
    "authenticate_client",
    description="Valida CPF e nascimento informados",
)
def authenticate_client(
    cpf: str,
    birth_date: str,
    state: InjectedState,
    service: InjectedAuthenticationService,
) -> AuthenticationResult:
    """Delega autenticacao ao servico de triagem."""
    return service.authenticate(state, cpf, birth_date)


@tool(
    "get_credit_limit",
    description="Consulta o limite do cliente autenticado",
)
def get_credit_limit(
    state: InjectedState,
    service: InjectedCreditService,
) -> CreditLimitResult:
    """Delega consulta de limite apos validar autorizacao."""
    _require_authenticated(state)
    return service.get_credit_limit(state)


@tool(
    "request_limit_increase",
    description="Registra e avalia o limite solicitado",
)
def request_limit_increase(
    new_limit: Decimal,
    state: InjectedState,
    service: InjectedCreditService,
) -> LimitIncreaseResult:
    """Delega solicitacao de aumento apos validar autorizacao."""
    _require_authenticated(state)
    return service.request_limit_increase(state, new_limit)


@tool(
    "update_credit_score",
    description="Calcula e atualiza o score após entrevista",
)
def update_credit_score(
    interview: CreditInterview,
    state: InjectedState,
    service: InjectedInterviewService,
) -> ScoreUpdateResult:
    """Delega atualizacao de score apos validar autorizacao."""
    _require_authenticated(state)
    return service.update_credit_score(state, interview)


@tool(
    "get_exchange_rate",
    description="Consulta a cotação atual de um par de moedas",
)
def get_exchange_rate(
    base_currency: str,
    quote_currency: str,
    state: InjectedState,
    service: InjectedExchangeService,
) -> ExchangeRateResult:
    """Delega consulta de cambio apos validar autorizacao."""
    _require_authenticated(state)
    return service.get_exchange_rate(state, base_currency, quote_currency)


@tool(
    "end_service",
    description="Encerra o atendimento atual",
)
def end_service(reason: EndReason) -> EndServiceResult:
    """Retorna sinal tipado de encerramento para a orquestracao."""
    return EndServiceResult(reason=reason)


def _require_authenticated(state: ConversationState) -> None:
    if state.authenticated_client is None:
        raise AuthorizationError("authenticated client is required")
