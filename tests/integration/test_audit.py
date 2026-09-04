"""Testes de integracao da auditoria tecnica e das metricas do LLM."""

import logging
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import TypeVar

import pytest
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ValidationError

from banco_agil.agents.graph import GraphDependencies, build_graph
from banco_agil.agents.state import ConversationState
from banco_agil.domain.enums import Agent, AuditEventType
from banco_agil.domain.exceptions import IntegrationError
from banco_agil.domain.models import AuditEvent, Client
from banco_agil.observability.logging import get_logger, sanitize_context
from banco_agil.observability.metrics import InMemoryLlmMetricsRecorder, LlmCallMetrics
from banco_agil.repositories.audit_sqlite import AuditSqliteRepository
from banco_agil.services.authentication import AuthenticationService
from banco_agil.services.conversation import ConversationService
from banco_agil.services.credit import CreditService
from banco_agil.services.credit_interview import CreditInterviewService
from banco_agil.services.exchange import ExchangeService

OutputModel = TypeVar("OutputModel", bound=BaseModel)
FAKE_CPF = "01234567890"
FAKE_BIRTH_DATE = "1990-05-20"


@dataclass
class MemoryClientRepository:
    """Mantem um cliente ficticio em memoria."""

    client: Client

    def find_by_cpf(self, cpf: str) -> Client | None:
        """Retorna o cliente quando o CPF coincide."""
        return self.client if cpf == self.client.cpf else None

    def update_credit_score(self, cpf: str, credit_score: int) -> Client:
        """Atualiza o cliente em memoria."""
        self.client = self.client.model_copy(update={"credit_score": credit_score})
        return self.client


@dataclass
class MemoryScoreLimitRepository:
    """Retorna limite maximo fixo para qualquer score."""

    def find_max_limit(self, score: int) -> Decimal:
        """Retorna o teto ficticio da faixa."""
        del score
        return Decimal("5000.00")


@dataclass
class MemoryCreditRequestRepository:
    """Registra solicitacoes sem persistencia real."""

    requests: list[object] = field(default_factory=list)

    def create(self, request: object) -> object:
        """Registra uma solicitacao pendente."""
        self.requests.append(request)
        return request

    def finalize(self, request: object, status: object) -> object:
        """Retorna a solicitacao recebida sem alteracao."""
        del status
        return request


@dataclass
class SilentExchangeProvider:
    """Provedor que nunca e chamado nos testes de auditoria."""

    def get_exchange_rate(self, base_currency: str, quote_currency: str) -> object:
        """Falha se alguma cotacao for solicitada."""
        raise AssertionError(f"unexpected quote {base_currency}-{quote_currency}")


@dataclass
class MetricsRecordingLlm:
    """Fake que classifica e registra metrica como o adaptador real."""

    response: object
    metrics: InMemoryLlmMetricsRecorder
    calls: list[str] = field(default_factory=list)

    def calls_remaining(self, turn_id: str) -> int:
        """Informa o saldo de chamadas do turno registrado."""
        used = sum(1 for call in self.calls if call == turn_id)
        return max(0, 2 - used)

    def invoke_structured(
        self,
        turn_id: str,
        messages: list[BaseMessage],
        output_schema: type[OutputModel],
        *,
        prompt_version: str | None = None,
    ) -> OutputModel:
        """Valida a resposta e registra modelo e versao do prompt."""
        self.calls.append(turn_id)
        self.metrics.record(
            LlmCallMetrics(
                model="fake-model",
                duration_ms=1.0,
                succeeded=True,
                prompt_version=prompt_version,
            )
        )
        try:
            return output_schema.model_validate(self.response)
        except ValidationError as error:
            raise IntegrationError("invalid structured LLM output") from error


def _client() -> Client:
    return Client(
        cpf=FAKE_CPF,
        birth_date=date(1990, 5, 20),
        credit_limit=Decimal("2500.00"),
        credit_score=700,
    )


def _service(
    audit: AuditSqliteRepository | None,
    llm: MetricsRecordingLlm | None = None,
    metrics: InMemoryLlmMetricsRecorder | None = None,
) -> ConversationService:
    clients = MemoryClientRepository(_client())
    del metrics
    dependencies = GraphDependencies(
        authentication=AuthenticationService(clients),
        credit=CreditService(
            MemoryScoreLimitRepository(), MemoryCreditRequestRepository()
        ),
        credit_interview=CreditInterviewService(clients),
        exchange=ExchangeService(SilentExchangeProvider()),
        llm=llm,
    )
    return ConversationService(
        build_graph(dependencies), lambda: "turn-id", audit_store=audit
    )


def _raw_rows(path: Path) -> str:
    with closing(sqlite3.connect(path)) as connection:
        rows = connection.execute("SELECT * FROM audit_events").fetchall()
    return str(rows)


def test_turn_events_allow_session_diagnosis(tmp_path: Path) -> None:
    audit = AuditSqliteRepository(tmp_path / "audit.db")
    service = _service(audit)
    state = ConversationState()

    service.handle_turn(state, (), FAKE_CPF, session_id="session-1")
    service.handle_turn(state, (), FAKE_BIRTH_DATE, session_id="session-1")

    events = audit.list_by_session("session-1")
    types = [event.event_type for event in events]
    assert types[0] is AuditEventType.STARTED
    assert types[-1] is AuditEventType.FINISHED
    assert all(event.session_id == "session-1" for event in events)
    completed = [
        event
        for event in events
        if event.event_type
        in {
            AuditEventType.TRANSITION,
            AuditEventType.ERROR,
            AuditEventType.FINISHED,
            AuditEventType.ENDED,
        }
    ]
    assert completed
    assert all(
        event.duration_ms is not None and event.duration_ms >= 0 for event in completed
    )
    assert {event.agent for event in events} <= set(Agent) | {None}
    assert state.authenticated


def test_audit_storage_and_logs_contain_no_pii(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    audit = AuditSqliteRepository(tmp_path / "audit.db")
    service = _service(audit)
    state = ConversationState(authenticated_client=_client())
    logger = get_logger("banco_agil.audit.test")

    with caplog.at_level(logging.INFO, logger=logger.name):
        service.handle_turn(
            state, (), f"meu CPF é {FAKE_CPF} e nasci em {FAKE_BIRTH_DATE}"
        )
        logger.info("manual check %s", sanitize_context({"cpf": FAKE_CPF}))

    dump = _raw_rows(tmp_path / "audit.db")
    assert FAKE_CPF not in dump
    assert FAKE_BIRTH_DATE not in dump
    assert "2500.00" not in dump
    assert FAKE_CPF not in caplog.text
    assert FAKE_BIRTH_DATE not in caplog.text


def test_metrics_distinguish_zero_and_bounded_llm_calls() -> None:
    metrics = InMemoryLlmMetricsRecorder()
    service = _service(None)
    state = ConversationState(authenticated_client=_client())

    service.handle_turn(state, (), "quero aumentar meu limite")
    assert metrics.calls == []

    llm = MetricsRecordingLlm(
        {
            "intent": "credit_limit",
            "reply": (
                "Com certeza! Seu limite atual é [DADO_1]. Posso ajudar em algo mais?"
            ),
        },
        metrics,
    )
    humanized_service = _service(None, llm)
    humanized_state = ConversationState(authenticated_client=_client())
    turn = humanized_service.handle_turn(
        humanized_state, (), "quero aumentar meu limite"
    )
    assert turn.reply.startswith("Com certeza!")
    assert len(metrics.calls) == 2
    assert [call.prompt_version for call in metrics.calls] == [
        "global@1.3.0+triage@1.2.0",
        "global@1.3.0+credit@1.2.0",
    ]
    assert all(call.model == "fake-model" for call in metrics.calls)

    ambiguous_metrics = InMemoryLlmMetricsRecorder()
    ambiguous_llm = MetricsRecordingLlm({"intent": "other"}, ambiguous_metrics)
    ambiguous_service = _service(None, ambiguous_llm)
    ambiguous_state = ConversationState(authenticated_client=_client())
    ambiguous_service.handle_turn(ambiguous_state, (), "preciso resolver outra coisa")
    assert len(ambiguous_metrics.calls) == 2
    assert all(
        call.prompt_version == "global@1.3.0+triage@1.2.0"
        for call in ambiguous_metrics.calls
    )


def test_audit_failure_is_non_fatal(tmp_path: Path) -> None:
    audit = AuditSqliteRepository(tmp_path / "missing-dir" / "audit.db")
    service = _service(audit)

    turn = service.handle_turn(ConversationState(), (), "oi")

    assert "CPF" in turn.reply


def test_agent_change_records_transition_event(tmp_path: Path) -> None:
    audit = AuditSqliteRepository(tmp_path / "audit.db")
    service = _service(audit)
    state = ConversationState(authenticated_client=_client())

    service.handle_turn(state, (), "quero aumentar meu limite", session_id="s-t")

    events = audit.list_by_session("s-t")
    transitions = [
        event for event in events if event.event_type is AuditEventType.TRANSITION
    ]
    assert len(transitions) == 1
    assert transitions[0].agent is Agent.CREDIT
    assert state.active_agent is Agent.CREDIT


def test_repository_round_trips_integration_event_with_llm_fields(
    tmp_path: Path,
) -> None:
    from datetime import UTC, datetime

    audit = AuditSqliteRepository(tmp_path / "audit.db")
    recorded = audit.record(
        AuditEvent(
            session_id="s-i",
            event_type=AuditEventType.INTEGRATION,
            agent=Agent.TRIAGE,
            result="ok",
            duration_ms=12.5,
            model="fake-model",
            prompt_version="global@1.3.0+triage@1.2.0",
            llm_calls=1,
            input_tokens=120,
            output_tokens=30,
            created_at=datetime(2026, 9, 3, 12, 0, tzinfo=UTC),
        )
    )

    assert audit.list_by_session("s-i") == [recorded]


def test_ended_turn_records_ended_event(tmp_path: Path) -> None:
    audit = AuditSqliteRepository(tmp_path / "audit.db")
    service = _service(audit)
    state = ConversationState(authenticated_client=_client())

    turn = service.handle_turn(state, (), "encerrar", session_id="session-end")

    assert "encerrado" in turn.reply.casefold()
    assert state.ended
    events = audit.list_by_session("session-end")
    assert events[-1].event_type is AuditEventType.ENDED


def test_sanitize_context_drops_sensitive_keys() -> None:
    sanitized = sanitize_context(
        {
            "active_agent": "triage",
            "authenticated": False,
            "cpf": FAKE_CPF,
            "birth_date": FAKE_BIRTH_DATE,
            "monthly_income": "5000.00",
            "user_text": "oi",
            "reply": "olá",
            "prompt": "secret instructions",
        }
    )

    assert sanitized == {"active_agent": "triage", "authenticated": False}
