"""Consulta e avaliacao deterministica de credito."""

from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation

from banco_agil.agents.state import ConversationState
from banco_agil.domain.enums import CreditRequestStatus
from banco_agil.domain.exceptions import AuthorizationError, DomainError
from banco_agil.domain.models import (
    Client,
    CreditLimitResult,
    CreditRequest,
    LimitIncreaseResult,
)
from banco_agil.repositories.protocols import (
    ClientRepository,
    CreditRequestRepository,
    ScoreLimitRepository,
)

CENT = Decimal("0.01")


def _utc_now() -> datetime:
    return datetime.now(UTC)


class CreditService:
    """Aplica regras de credito sem depender de LLM."""

    def __init__(
        self,
        score_limit_repository: ScoreLimitRepository,
        credit_request_repository: CreditRequestRepository,
        client_repository: ClientRepository,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        """Recebe repositorios e relogio injetavel para avaliacao."""
        self._score_limit_repository = score_limit_repository
        self._credit_request_repository = credit_request_repository
        self._client_repository = client_repository
        self._clock = clock

    def get_credit_limit(self, state: ConversationState) -> CreditLimitResult:
        """Retorna o limite do cliente autenticado e confiavel.

        Raises:
            AuthorizationError: Se a sessao nao estiver autenticada.
        """
        client = self._require_authenticated_client(state)
        return CreditLimitResult(current_limit=client.credit_limit)

    def request_limit_increase(
        self,
        state: ConversationState,
        new_limit: Decimal,
    ) -> LimitIncreaseResult:
        """Registra, avalia e finaliza uma solicitacao de aumento.

        Raises:
            AuthorizationError: Se a sessao nao estiver autenticada.
            DomainError: Se o novo limite nao for finito e maior que o atual.
        """
        client = self._require_authenticated_client(state)
        try:
            normalized_limit = new_limit.quantize(CENT)
        except InvalidOperation as error:
            raise DomainError(
                "new limit must be finite and greater than current limit"
            ) from error
        if not new_limit.is_finite() or normalized_limit <= client.credit_limit:
            raise DomainError("new limit must be finite and greater than current limit")

        pending_request = CreditRequest(
            client_cpf=client.cpf,
            requested_at=self._clock(),
            current_limit=client.credit_limit,
            requested_limit=normalized_limit,
            status=CreditRequestStatus.PENDING,
        )
        created_request = self._credit_request_repository.create(pending_request)
        maximum_limit = self._score_limit_repository.find_max_limit(client.credit_score)
        status = (
            CreditRequestStatus.APPROVED
            if created_request.requested_limit <= maximum_limit
            else CreditRequestStatus.REJECTED
        )
        finalized_request = self._credit_request_repository.finalize(
            created_request,
            status,
        )
        if status is CreditRequestStatus.REJECTED:
            state.requested_limit = finalized_request.requested_limit
            return LimitIncreaseResult(
                current_limit=finalized_request.current_limit,
                requested_limit=finalized_request.requested_limit,
                status=finalized_request.status,
                offer_interview=True,
            )

        updated_client = self._client_repository.update_credit_limit(
            client.cpf,
            finalized_request.requested_limit,
        )
        state.authenticated_client = updated_client
        state.requested_limit = None
        return LimitIncreaseResult(
            current_limit=updated_client.credit_limit,
            requested_limit=finalized_request.requested_limit,
            status=finalized_request.status,
            offer_interview=False,
        )

    @staticmethod
    def _require_authenticated_client(state: ConversationState) -> Client:
        client = state.authenticated_client
        if client is None:
            raise AuthorizationError("authenticated client is required")
        return client
