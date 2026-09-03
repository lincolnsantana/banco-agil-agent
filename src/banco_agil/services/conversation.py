"""Servico de aplicacao para executar um turno conversacional."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast
from uuid import uuid4

from langchain_core.messages import BaseMessage, HumanMessage

from banco_agil.agents.graph import ConversationGraph
from banco_agil.agents.router import GraphState
from banco_agil.agents.state import ConversationState
from banco_agil.domain.exceptions import DomainError

_GRAPH_RECURSION_LIMIT = 8
_HISTORY_MAX_MESSAGES = 6


def _new_turn_id() -> str:
    return uuid4().hex


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
    ) -> None:
        """Recebe grafo compilado e gerador de identificadores injetavel."""
        self.graph = graph
        self._turn_id_factory = turn_id_factory

    def handle_turn(
        self,
        state: ConversationState,
        history: Sequence[BaseMessage],
        user_text: str,
    ) -> ConversationTurn:
        """Executa um turno e retorna no maximo seis mensagens recentes.

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

        graph_input: GraphState = {
            "conversation": state,
            "messages": [
                *history[-_HISTORY_MAX_MESSAGES:],
                HumanMessage(content=normalized_text),
            ],
            "user_text": normalized_text,
            "turn_id": turn_id,
            "reply": "",
            "step_count": 0,
        }
        result = cast(
            GraphState,
            self.graph.invoke(
                graph_input,
                config={"recursion_limit": _GRAPH_RECURSION_LIMIT},
            ),
        )
        return ConversationTurn(
            state=result["conversation"],
            history=tuple(result["messages"]),
            reply=result["reply"],
        )
