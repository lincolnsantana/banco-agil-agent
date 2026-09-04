"""Coleta validada e incremental da entrevista de credito."""

import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum

from banco_agil.agents.state import ConversationState, CreditInterviewDraft
from banco_agil.domain.enums import Agent, EmploymentType
from banco_agil.domain.exceptions import AuthorizationError, DomainError
from banco_agil.domain.models import Client, CreditInterview, ScoreUpdateResult
from banco_agil.repositories.protocols import ClientRepository
from banco_agil.services.score import calculate_credit_score


class InterviewField(StrEnum):
    """Campo esperado na proxima etapa da entrevista."""

    MONTHLY_INCOME = "monthly_income"
    EMPLOYMENT_TYPE = "employment_type"
    MONTHLY_EXPENSES = "monthly_expenses"
    DEPENDENTS = "dependents"
    ACTIVE_DEBTS = "active_debts"


@dataclass(frozen=True)
class InterviewProgress:
    """Progresso seguro retornado a cada etapa da entrevista."""

    next_field: InterviewField | None
    score_update: ScoreUpdateResult | None = None
    completed_interview: CreditInterview | None = None
    consent_declined: bool = False

    @property
    def completed(self) -> bool:
        """Informa se o score foi atualizado ao concluir a entrevista."""
        return self.score_update is not None


class CreditInterviewService:
    """Coordena consentimento, respostas e atualizacao final do score."""

    def __init__(self, client_repository: ClientRepository) -> None:
        """Recebe o repositorio de clientes por protocolo."""
        self._client_repository = client_repository

    def start(
        self,
        state: ConversationState,
        consent: bool,
    ) -> InterviewProgress:
        """Inicia a coleta com autenticacao e consentimento explicito.

        Aceita entrevista direta (revisao de score) ou apos limite rejeitado.

        Raises:
            AuthorizationError: Se nao houver cliente autenticado.
        """
        self._require_authenticated_client(state)
        state.interview_draft = CreditInterviewDraft()
        if not consent:
            return InterviewProgress(next_field=None, consent_declined=True)

        state.interview_draft = CreditInterviewDraft(consent_given=True)
        state.active_agent = Agent.CREDIT_INTERVIEW
        return InterviewProgress(next_field=InterviewField.MONTHLY_INCOME)

    def submit_answer(
        self,
        state: ConversationState,
        answer: str,
    ) -> InterviewProgress:
        """Valida uma resposta e avanca exatamente um campo.

        Raises:
            AuthorizationError: Se nao houver cliente autenticado.
            DomainError: Se faltar consentimento ou a resposta for invalida.
        """
        progress = self.collect_answer(state, answer)
        if progress.completed_interview is None:
            return progress
        score_update = self.update_credit_score(state, progress.completed_interview)
        return InterviewProgress(next_field=None, score_update=score_update)

    def collect_answer(
        self,
        state: ConversationState,
        answer: str,
    ) -> InterviewProgress:
        """Valida uma resposta sem persistir o score final.

        Raises:
            AuthorizationError: Se nao houver cliente autenticado.
            DomainError: Se faltar consentimento ou a resposta for invalida.
        """
        self._require_authenticated_client(state)
        draft = state.interview_draft
        if not draft.consent_given:
            raise DomainError("credit interview requires consent")

        current_field = self._next_field(draft)
        if current_field is None:
            raise DomainError("credit interview is already complete")

        try:
            if current_field is InterviewField.MONTHLY_INCOME:
                draft.monthly_income = _parse_money(answer)
            elif current_field is InterviewField.EMPLOYMENT_TYPE:
                draft.employment_type = _parse_employment_type(answer)
            elif current_field is InterviewField.MONTHLY_EXPENSES:
                draft.monthly_expenses = _parse_money(answer)
            elif current_field is InterviewField.DEPENDENTS:
                draft.dependents = _parse_dependents(answer)
            else:
                return self._complete_interview(state, _parse_active_debts(answer))
        except (InvalidOperation, ValueError) as error:
            raise DomainError("invalid interview answer") from error

        return InterviewProgress(next_field=self._next_field(draft))

    def update_credit_score(
        self,
        state: ConversationState,
        interview: CreditInterview,
    ) -> ScoreUpdateResult:
        """Calcula e persiste o score de uma entrevista completa e validada.

        Quando houver limite rejeitado, agenda a reanalise no credito; em
        entrevista direta, apenas conclui com o novo score.

        Raises:
            AuthorizationError: Se nao houver cliente autenticado.
        """
        client = self._require_authenticated_client(state)
        new_score = calculate_credit_score(interview)
        updated_client = self._client_repository.update_credit_score(
            client.cpf,
            new_score,
        )
        state.authenticated_client = updated_client
        state.interview_draft = CreditInterviewDraft()
        if state.requested_limit is None:
            state.credit_reanalysis_pending = False
            state.active_agent = Agent.TRIAGE
        else:
            state.credit_reanalysis_pending = True
            state.active_agent = Agent.CREDIT
        return ScoreUpdateResult(
            previous_score=client.credit_score,
            new_score=updated_client.credit_score,
        )

    def _complete_interview(
        self,
        state: ConversationState,
        has_active_debts: bool,
    ) -> InterviewProgress:
        interview = CreditInterview.model_validate(
            state.interview_draft.model_dump() | {"has_active_debts": has_active_debts}
        )
        return InterviewProgress(
            next_field=None,
            completed_interview=interview,
        )

    @staticmethod
    def _require_authenticated_client(state: ConversationState) -> Client:
        client = state.authenticated_client
        if client is None:
            raise AuthorizationError("authenticated client is required")
        return client

    @staticmethod
    def _next_field(draft: CreditInterviewDraft) -> InterviewField | None:
        if draft.monthly_income is None:
            return InterviewField.MONTHLY_INCOME
        if draft.employment_type is None:
            return InterviewField.EMPLOYMENT_TYPE
        if draft.monthly_expenses is None:
            return InterviewField.MONTHLY_EXPENSES
        if draft.dependents is None:
            return InterviewField.DEPENDENTS
        if draft.has_active_debts is None:
            return InterviewField.ACTIVE_DEBTS
        return None


def _parse_money(answer: str) -> Decimal:
    value = Decimal(answer.strip())
    if not value.is_finite() or value < 0:
        raise ValueError("money must be finite and non-negative")
    return value


_SHORT_ANSWER_PUNCTUATION = ".,!?;:'\"()[]-"

_EMPLOYMENT_BY_NORMALIZED = {
    "formal": EmploymentType.FORMAL,
    "autonomo": EmploymentType.SELF_EMPLOYED,
    "desempregado": EmploymentType.UNEMPLOYED,
}


def _normalize_short_answer(answer: str) -> str:
    """Remove acentos, caixa e pontuação lateral de respostas curtas."""
    without_accents = "".join(
        character
        for character in unicodedata.normalize("NFKD", answer.casefold().strip())
        if not unicodedata.combining(character)
    )
    return " ".join(without_accents.split()).strip(_SHORT_ANSWER_PUNCTUATION).strip()


def _parse_employment_type(answer: str) -> EmploymentType:
    try:
        return _EMPLOYMENT_BY_NORMALIZED[_normalize_short_answer(answer)]
    except KeyError as error:
        raise ValueError(
            "employment type must be formal, autônomo ou desempregado"
        ) from error


def _parse_dependents(answer: str) -> int:
    cleaned = answer.strip().rstrip(_SHORT_ANSWER_PUNCTUATION).strip()
    value = int(cleaned)
    if value < 0:
        raise ValueError("dependents cannot be negative")
    return value


def _parse_active_debts(answer: str) -> bool:
    normalized = _normalize_short_answer(answer)
    if normalized == "sim":
        return True
    if normalized == "nao":
        return False
    raise ValueError("active debts answer must be sim or não")
