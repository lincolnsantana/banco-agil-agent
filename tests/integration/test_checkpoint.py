"""Testes de integracao do checkpoint opcional de sessoes."""

import itertools
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path

from langchain_core.messages import BaseMessage

from banco_agil.agents.graph import (
    GraphDependencies,
    SessionCheckpointStore,
    build_graph,
)
from banco_agil.agents.state import ConversationState
from banco_agil.config import Settings
from banco_agil.domain.exceptions import RepositoryError
from banco_agil.domain.models import Client
from banco_agil.services.authentication import AuthenticationService
from banco_agil.services.conversation import ConversationService
from banco_agil.services.credit import CreditService
from banco_agil.services.credit_interview import CreditInterviewService
from banco_agil.services.exchange import ExchangeService


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
    """Provedor que nunca e chamado nos testes de checkpoint."""

    def get_exchange_rate(self, base_currency: str, quote_currency: str) -> object:
        """Falha se alguma cotacao for solicitada."""
        raise AssertionError(f"unexpected quote {base_currency}-{quote_currency}")


def _service(store: SessionCheckpointStore | None) -> ConversationService:
    clients = MemoryClientRepository(
        Client(
            cpf="01234567890",
            birth_date=date(1990, 5, 20),
            credit_limit=Decimal("2500.00"),
            credit_score=700,
        )
    )
    dependencies = GraphDependencies(
        authentication=AuthenticationService(clients),
        credit=CreditService(
            MemoryScoreLimitRepository(), MemoryCreditRequestRepository()
        ),
        credit_interview=CreditInterviewService(clients),
        exchange=ExchangeService(SilentExchangeProvider()),
    )
    counter = itertools.count(1)
    return ConversationService(
        build_graph(dependencies),
        lambda: f"checkpoint-turn-{next(counter)}",
        checkpoint_store=store,
    )


def test_restart_recovers_authenticated_session(tmp_path: Path) -> None:
    store = SessionCheckpointStore(tmp_path / "checkpoints.db")
    service = _service(store)
    state = ConversationState()
    history: tuple[BaseMessage, ...] = ()
    for text in ("01234567890", "1990-05-20"):
        turn = service.handle_turn(state, history, text, session_id="sessao-1")
        history = turn.history
    assert state.authenticated

    restarted = _service(SessionCheckpointStore(tmp_path / "checkpoints.db"))
    restored = restarted.load_session("sessao-1")

    assert restored is not None
    restored_state, restored_history = restored
    assert restored_state.authenticated
    assert restored_state.authentication_attempts == 0
    assert [message.content for message in restored_history] == [
        message.content for message in history
    ]
    continued = restarted.load_session("sessao-1")
    assert continued is not None
    reply = restarted.handle_turn(continued[0], continued[1], "qual é meu limite?")
    assert "2.500,00" in reply.reply


def test_flows_work_without_checkpoint() -> None:
    service = _service(None)

    turn = service.handle_turn(ConversationState(), (), "oi")

    assert "CPF" in turn.reply
    assert service.load_session("qualquer") is None


def test_checkpoint_failure_is_non_fatal(tmp_path: Path) -> None:
    service = _service(SessionCheckpointStore(tmp_path / "missing-dir" / "c.db"))

    turn = service.handle_turn(ConversationState(), (), "oi", session_id="sessao-falha")

    assert "CPF" in turn.reply


def test_checkpoint_stores_only_necessary_data(tmp_path: Path) -> None:
    path = tmp_path / "checkpoints.db"
    service = _service(SessionCheckpointStore(path))
    state = ConversationState()
    service.handle_turn(state, (), "01234567890", session_id="sessao-2")

    with closing(sqlite3.connect(path)) as connection:
        columns = [
            row[1]
            for row in connection.execute("PRAGMA table_info(checkpoints)").fetchall()
        ]
        dump = str(connection.execute("SELECT * FROM checkpoints").fetchall())

    assert columns == ["session_id", "conversation", "history", "updated_at"]
    assert "user_text" not in dump
    assert "turn_id" not in dump


def test_clear_session_removes_checkpoint(tmp_path: Path) -> None:
    store = SessionCheckpointStore(tmp_path / "checkpoints.db")
    service = _service(store)
    service.handle_turn(ConversationState(), (), "oi", session_id="sessao-3")
    assert service.load_session("sessao-3") is not None

    service.clear_session("sessao-3")

    assert service.load_session("sessao-3") is None


def test_corrupt_checkpoint_raises_controlled_error(tmp_path: Path) -> None:
    path = tmp_path / "checkpoints.db"
    service = _service(SessionCheckpointStore(path))
    service.handle_turn(ConversationState(), (), "oi", session_id="sessao-4")
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            "UPDATE checkpoints SET conversation = 'nao-e-json' WHERE session_id = ?",
            ("sessao-4",),
        )
        connection.commit()

    try:
        service.load_session("sessao-4")
    except RepositoryError:
        return
    raise AssertionError("corrupt checkpoint should raise RepositoryError")


def test_checkpoint_path_is_optional_and_configurable(tmp_path: Path) -> None:
    assert Settings().conversation_checkpoint_path is None

    settings = Settings(conversation_checkpoint_path=tmp_path / "checkpoints.db")

    assert settings.conversation_checkpoint_path == tmp_path / "checkpoints.db"
