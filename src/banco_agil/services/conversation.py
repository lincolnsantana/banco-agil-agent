"""Servico de aplicacao para executar um turno conversacional."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import cast
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from banco_agil.agents.graph import ConversationGraph, SessionCheckpointStore
from banco_agil.agents.router import GraphState
from banco_agil.agents.state import ConversationState
from banco_agil.domain.enums import Agent, AuditEventType
from banco_agil.domain.exceptions import DomainError
from banco_agil.domain.models import AuditEvent
from banco_agil.observability.logging import get_logger, sanitize_context
from banco_agil.repositories.audit_sqlite import AuditSqliteRepository

_GRAPH_RECURSION_LIMIT = 8
_HISTORY_MAX_MESSAGES = 6

_logger = get_logger("banco_agil.conversation")


def _new_turn_id() -> str:
    return uuid4().hex


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class ConversationTurn:
    """Resultado observavel de um turno completo do grafo."""

    state: ConversationState
    history: tuple[BaseMessage, ...]
    reply: str


class ConversationService:
    """Executa um turno preservando estado e historico fornecidos pela UI."""

    def __init__(
        self,
        graph: ConversationGraph,
        turn_id_factory: Callable[[], str] = _new_turn_id,
        *,
        audit_store: AuditSqliteRepository | None = None,
        checkpoint_store: SessionCheckpointStore | None = None,
    ) -> None:
        """Recebe grafo compilado, gerador de IDs e integracoes opcionais."""
        self.graph = graph
        self._turn_id_factory = turn_id_factory
        self._audit_store = audit_store
        self._checkpoint_store = checkpoint_store

    def handle_turn(
        self,
        state: ConversationState,
        history: Sequence[BaseMessage],
        user_text: str,
        *,
        session_id: str | None = None,
    ) -> ConversationTurn:
        """Executa um turno e retorna no maximo seis mensagens recentes.

        A auditoria, quando configurada, nunca altera o resultado do turno.

        Raises:
            DomainError: Se a conversa ja estiver encerrada.
            ValueError: Se mensagem ou identificador do turno forem vazios.
        """
        normalized_text = user_text.strip()
        if not normalized_text:
            raise ValueError("message cannot be empty")
        if state.ended:
            raise DomainError("conversation is already ended")
        turn_id = self._turn_id_factory()
        if not turn_id:
            raise ValueError("turn ID cannot be empty")

        event_session_id = session_id or turn_id
        from_agent = state.active_agent
        started_at = perf_counter()
        self._record(
            event_session_id, AuditEventType.STARTED, from_agent, "started", None
        )

        graph_input: GraphState = {
            "conversation": state,
            "messages": [
                *history[-_HISTORY_MAX_MESSAGES:],
                HumanMessage(content=normalized_text),
            ],
            "user_text": normalized_text,
            "turn_id": turn_id,
            "reply": "",
            "responding_agent": None,
            "step_count": 0,
            "context": None,
        }
        try:
            result = cast(
                GraphState,
                self.graph.invoke(
                    graph_input,
                    config={"recursion_limit": _GRAPH_RECURSION_LIMIT},
                ),
            )
        except Exception:
            self._record(
                event_session_id,
                AuditEventType.ERROR,
                from_agent,
                "error",
                (perf_counter() - started_at) * 1000,
            )
            raise
        duration_ms = (perf_counter() - started_at) * 1000
        to_agent = result["conversation"].active_agent
        if to_agent is not from_agent:
            self._record(
                event_session_id,
                AuditEventType.TRANSITION,
                to_agent,
                "transitioned",
                duration_ms,
            )
        if result["conversation"].ended:
            final_type = AuditEventType.ENDED
            final_result = "ended"
        else:
            final_type = AuditEventType.FINISHED
            final_result = "ok"
        self._record(event_session_id, final_type, to_agent, final_result, duration_ms)
        if session_id is not None:
            self._save_checkpoint(
                session_id, result["conversation"], result["messages"]
            )
        _logger.info(
            "turn finished",
            extra={
                "audit": sanitize_context(
                    {
                        **state.to_log_context(),
                        "session_id": event_session_id,
                        "duration_ms": round(duration_ms, 3),
                    }
                )
            },
        )
        display_history = (
            *history,
            HumanMessage(content=normalized_text),
            AIMessage(content=result["reply"]),
        )
        return ConversationTurn(
            state=result["conversation"],
            history=display_history,
            reply=result["reply"],
        )

    def load_session(
        self, session_id: str
    ) -> tuple[ConversationState, list[BaseMessage]] | None:
        """Restaura estado e historico validos, ou None quando ausentes.

        Sem store configurado, sempre retorna None. Checkpoints corrompidos
        geram `RepositoryError` em vez de estado parcial.

        Raises:
            RepositoryError: Se o checkpoint existir mas for invalido.
        """
        if self._checkpoint_store is None:
            return None
        return self._checkpoint_store.load(session_id)

    def clear_session(self, session_id: str) -> None:
        """Remove o checkpoint da sessao; sem store, nao faz nada.

        Raises:
            RepositoryError: Se o banco nao puder ser atualizado.
        """
        if self._checkpoint_store is None:
            return
        self._checkpoint_store.clear(session_id)

    def _save_checkpoint(
        self,
        session_id: str,
        state: ConversationState,
        history: Sequence[BaseMessage],
    ) -> None:
        """Persiste o checkpoint sem permitir que falhas afetem o turno."""
        if self._checkpoint_store is None:
            return
        try:
            self._checkpoint_store.save(session_id, state, history)
        except Exception:
            _logger.warning("session checkpoint failed")

    def _record(
        self,
        session_id: str,
        event_type: AuditEventType,
        agent: Agent | None,
        result: str,
        duration_ms: float | None,
    ) -> None:
        """Persiste um evento sem permitir que falhas afetem o turno."""
        if self._audit_store is None:
            return
        try:
            self._audit_store.record(
                AuditEvent(
                    session_id=session_id,
                    event_type=event_type,
                    agent=agent,
                    result=result,
                    duration_ms=duration_ms,
                    created_at=_utc_now(),
                )
            )
        except Exception:
            _logger.warning("audit record failed")
