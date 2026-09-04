"""Estado validado compartilhado durante uma conversa."""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator

from banco_agil.domain.enums import Agent, EmploymentType, EndReason, Intent
from banco_agil.domain.models import Client, Cpf, Money, PositiveMoney


class CreditInterviewDraft(BaseModel):
    """Dados parciais da entrevista, armazenados somente apos consentimento."""

    model_config = ConfigDict(validate_assignment=True)

    consent_given: bool = False
    monthly_income: Money | None = Field(default=None, repr=False)
    employment_type: EmploymentType | None = Field(default=None, repr=False)
    monthly_expenses: Money | None = Field(default=None, repr=False)
    dependents: int | None = Field(default=None, ge=0, repr=False)
    has_active_debts: bool | None = Field(default=None, repr=False)

    @model_validator(mode="after")
    def require_consent_for_answers(self) -> "CreditInterviewDraft":
        """Impede armazenamento de respostas antes do consentimento."""
        answers = (
            self.monthly_income,
            self.employment_type,
            self.monthly_expenses,
            self.dependents,
            self.has_active_debts,
        )
        if not self.consent_given and any(answer is not None for answer in answers):
            raise ValueError("interview answers require consent")
        return self


class ConversationState(BaseModel):
    """Estado mutavel e seguro de uma sessao de atendimento."""

    model_config = ConfigDict(validate_assignment=True)

    authentication_attempts: int = Field(default=0, ge=0, le=3)
    active_agent: Agent = Agent.TRIAGE
    intent: Intent = Intent.UNKNOWN
    pending_cpf: Cpf | None = Field(default=None, repr=False)
    pending_birth_date: date | None = Field(default=None, repr=False)
    authenticated_client: Client | None = Field(default=None, repr=False)
    requested_limit: PositiveMoney | None = Field(default=None, repr=False)
    pending_flow: Intent | None = None
    credit_reanalysis_pending: bool = False
    interview_draft: CreditInterviewDraft = Field(
        default_factory=CreditInterviewDraft,
        repr=False,
    )
    ended: bool = False
    end_reason: EndReason | None = None

    @property
    def authenticated(self) -> bool:
        """Informa se a sessao possui um cliente confiavel."""
        return self.authenticated_client is not None

    @model_validator(mode="after")
    def validate_ending(self) -> "ConversationState":
        """Mantem o sinal de encerramento consistente com seu motivo."""
        if self.ended != (self.end_reason is not None):
            raise ValueError("ended state and end reason must be set together")
        return self

    def to_log_context(self) -> dict[str, str | int | bool | None]:
        """Retorna contexto operacional sem CPF ou dados financeiros."""
        return {
            "authentication_attempts": self.authentication_attempts,
            "active_agent": self.active_agent.value,
            "intent": self.intent.value,
            "pending_flow": self.pending_flow.value if self.pending_flow else None,
            "authenticated": self.authenticated,
            "ended": self.ended,
            "end_reason": self.end_reason.value if self.end_reason else None,
        }

    def end(self, reason: EndReason) -> None:
        """Marca o atendimento como encerrado preservando a invariante do estado."""
        validated = type(self).model_validate(
            self.model_dump() | {"ended": True, "end_reason": reason}
        )
        object.__setattr__(self, "ended", validated.ended)
        object.__setattr__(self, "end_reason", validated.end_reason)
