"""Modelos validados usados entre as camadas da aplicacao."""

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from banco_agil.domain.enums import (
    Agent,
    AuditEventType,
    CreditRequestStatus,
    EmploymentType,
    EndReason,
)

Cpf = Annotated[str, Field(pattern=r"^\d{11}$")]
Money = Annotated[Decimal, Field(ge=Decimal("0"), allow_inf_nan=False)]
PositiveMoney = Annotated[Decimal, Field(gt=Decimal("0"), allow_inf_nan=False)]
CreditScore = Annotated[int, Field(ge=0, le=1000)]


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return value.astimezone(UTC)


class DomainModel(BaseModel):
    """Base imutavel para contratos validados do dominio."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)


class Client(DomainModel):
    """Cliente carregado de uma fonte confiavel de dados."""

    cpf: Cpf = Field(repr=False)
    birth_date: date = Field(repr=False)
    credit_limit: Money
    credit_score: CreditScore

    @field_validator("birth_date")
    @classmethod
    def validate_birth_date(cls, value: date) -> date:
        """Rejeita datas de nascimento futuras."""
        if value > date.today():
            raise ValueError("birth date cannot be in the future")
        return value


class CreditRequest(DomainModel):
    """Solicitacao de aumento de limite com timestamp UTC."""

    client_cpf: Cpf = Field(repr=False)
    requested_at: datetime
    current_limit: Money
    requested_limit: PositiveMoney
    status: CreditRequestStatus

    @field_validator("requested_at")
    @classmethod
    def normalize_requested_at(cls, value: datetime) -> datetime:
        """Normaliza timestamps conscientes para UTC."""
        return _as_utc(value)


class CreditInterview(DomainModel):
    """Respostas completas e validas da entrevista de credito."""

    monthly_income: Money = Field(repr=False)
    employment_type: EmploymentType = Field(repr=False)
    monthly_expenses: Money = Field(repr=False)
    dependents: Annotated[int, Field(ge=0)] = Field(repr=False)
    has_active_debts: bool = Field(repr=False)


class ExchangeQuote(DomainModel):
    """Cotacao confirmada por uma fonte externa em um instante UTC."""

    base_currency: Annotated[str, Field(pattern=r"^[A-Z]{3}$")]
    quote_currency: Annotated[str, Field(pattern=r"^[A-Z]{3}$")]
    rate: PositiveMoney
    source: Annotated[str, Field(min_length=1)]
    quoted_at: datetime

    @field_validator("base_currency", "quote_currency", mode="before")
    @classmethod
    def normalize_currency(cls, value: object) -> object:
        """Normaliza codigos de moeda textuais para letras maiusculas."""
        return value.upper() if isinstance(value, str) else value

    @field_validator("quoted_at")
    @classmethod
    def normalize_quoted_at(cls, value: datetime) -> datetime:
        """Normaliza timestamps conscientes para UTC."""
        return _as_utc(value)


class AuthenticationResult(DomainModel):
    """Resultado seguro da tool de autenticacao."""

    authenticated: bool
    attempts: Annotated[int, Field(ge=0, le=3)]
    should_end: bool
    client: Client | None = Field(default=None, repr=False)

    @model_validator(mode="after")
    def validate_client_presence(self) -> "AuthenticationResult":
        """Exige cliente confiavel somente em autenticacao bem-sucedida."""
        if self.authenticated != (self.client is not None):
            raise ValueError("authenticated result must contain exactly one client")
        return self


class CreditLimitResult(DomainModel):
    """Resultado da consulta do limite de credito atual."""

    current_limit: Money


class LimitIncreaseResult(DomainModel):
    """Resultado da avaliacao de aumento de limite."""

    current_limit: Money
    requested_limit: PositiveMoney
    status: CreditRequestStatus
    offer_interview: bool = False


class ScoreUpdateResult(DomainModel):
    """Resultado da atualizacao deterministica do score de credito."""

    previous_score: CreditScore
    new_score: CreditScore


class ExchangeRateResult(DomainModel):
    """Resultado da consulta de cambio."""

    quote: ExchangeQuote


class EndServiceResult(DomainModel):
    """Resultado da tool que encerra o atendimento."""

    ended: Literal[True] = True
    reason: EndReason


class AuditEvent(DomainModel):
    """Evento tecnico de auditoria, sem texto do usuario ou resposta."""

    session_id: Annotated[str, Field(min_length=1)]
    event_type: AuditEventType
    agent: Agent | None = None
    result: Annotated[str, Field(min_length=1)]
    duration_ms: Annotated[float, Field(ge=0)] | None = None
    model: Annotated[str, Field(min_length=1)] | None = None
    prompt_version: Annotated[str, Field(min_length=1)] | None = None
    llm_calls: Annotated[int, Field(ge=0)] | None = None
    input_tokens: Annotated[int, Field(ge=0)] | None = None
    output_tokens: Annotated[int, Field(ge=0)] | None = None
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        """Normaliza timestamps conscientes para UTC."""
        return _as_utc(value)
