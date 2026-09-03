"""Auditoria tecnica em SQLite, sem dados pessoais ou financeiros."""

import sqlite3
from pathlib import Path
from typing import cast

from banco_agil.domain.enums import Agent, AuditEventType
from banco_agil.domain.exceptions import RepositoryError
from banco_agil.domain.models import AuditEvent

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    agent TEXT,
    result TEXT NOT NULL,
    duration_ms REAL,
    model TEXT,
    prompt_version TEXT,
    llm_calls INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    created_at TEXT NOT NULL
)
"""
_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_audit_events_session
ON audit_events (session_id)
"""
_COLUMNS = (
    "session_id",
    "event_type",
    "agent",
    "result",
    "duration_ms",
    "model",
    "prompt_version",
    "llm_calls",
    "input_tokens",
    "output_tokens",
    "created_at",
)


class AuditSqliteRepository:
    """Persiste eventos tecnicos de sessao em um SQLite local."""

    def __init__(self, path: Path) -> None:
        """Configura o caminho do banco sem realizar acesso imediato."""
        self._path = path

    def record(self, event: AuditEvent) -> AuditEvent:
        """Persiste um evento e o devolve.

        Raises:
            RepositoryError: Se o banco nao puder ser atualizado.
        """
        try:
            with sqlite3.connect(self._path) as connection:
                connection.execute(_CREATE_TABLE)
                connection.execute(_CREATE_INDEX)
                connection.execute(
                    "INSERT INTO audit_events"
                    " (session_id, event_type, agent, result, duration_ms,"
                    " model, prompt_version, llm_calls,"
                    " input_tokens, output_tokens, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    _serialize(event),
                )
                connection.commit()
        except (OSError, sqlite3.Error) as error:
            raise RepositoryError("audit events could not be recorded") from error
        return event

    def list_by_session(self, session_id: str) -> list[AuditEvent]:
        """Retorna os eventos de uma sessao na ordem de registro.

        Raises:
            RepositoryError: Se o banco nao puder ser lido.
        """
        try:
            with sqlite3.connect(self._path) as connection:
                connection.row_factory = sqlite3.Row
                rows = connection.execute(
                    "SELECT"
                    " session_id, event_type, agent, result, duration_ms,"
                    " model, prompt_version, llm_calls,"
                    " input_tokens, output_tokens, created_at"
                    " FROM audit_events WHERE session_id = ? ORDER BY id",
                    (session_id,),
                ).fetchall()
        except (OSError, sqlite3.Error) as error:
            raise RepositoryError("audit events could not be read") from error
        return [_deserialize(dict(row)) for row in rows]


def _serialize(event: AuditEvent) -> tuple[object, ...]:
    return (
        event.session_id,
        event.event_type.value,
        event.agent.value if event.agent is not None else None,
        event.result,
        event.duration_ms,
        event.model,
        event.prompt_version,
        event.llm_calls,
        event.input_tokens,
        event.output_tokens,
        event.created_at.isoformat(),
    )


def _deserialize(row: dict[str, object]) -> AuditEvent:
    data = dict(row)
    event_type = data.get("event_type")
    agent = data.get("agent")
    data["event_type"] = (
        AuditEventType(cast(str, event_type)) if event_type is not None else None
    )
    data["agent"] = Agent(cast(str, agent)) if agent is not None else None
    return AuditEvent.model_validate(data)
