"""Montagem do LangGraph com dependencias bancarias injetadas."""

import json
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    messages_from_dict,
    messages_to_dict,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

from banco_agil.agents._shared import HANDOFF_REPLY, humanize_reply
from banco_agil.agents.credit import handle_credit
from banco_agil.agents.credit_interview import handle_credit_interview
from banco_agil.agents.exchange import handle_exchange
from banco_agil.agents.knowledge import handle_knowledge
from banco_agil.agents.router import (
    GraphState,
    GraphUpdate,
    route_after_interview,
    route_after_specialist,
    route_after_triage,
    route_entry,
)
from banco_agil.agents.state import ConversationState
from banco_agil.agents.triage import handle_triage
from banco_agil.agents.understanding import TurnContext
from banco_agil.domain.enums import Agent
from banco_agil.domain.exceptions import RepositoryError
from banco_agil.integrations.llm import StructuredLlm
from banco_agil.services.authentication import AuthenticationService
from banco_agil.services.credit import CreditService
from banco_agil.services.credit_interview import CreditInterviewService
from banco_agil.services.exchange import ExchangeService
from banco_agil.services.knowledge import KnowledgeService

ConversationGraph = CompiledStateGraph[GraphState, None, GraphState, GraphState]

_CREATE_CHECKPOINTS = """
CREATE TABLE IF NOT EXISTS checkpoints (
    session_id TEXT PRIMARY KEY,
    conversation TEXT NOT NULL,
    history TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


@dataclass(frozen=True)
class GraphDependencies:
    """Servicos permitidos aos nos especialistas."""

    authentication: AuthenticationService
    credit: CreditService
    credit_interview: CreditInterviewService
    exchange: ExchangeService
    knowledge: KnowledgeService = field(default_factory=KnowledgeService)
    llm: StructuredLlm | None = None


class SessionCheckpointStore:
    """Persiste conversa e historico recentes por sessao em SQLite local.

    Guarda somente o necessario para continuar o atendimento apos reinicio:
    o estado validado e as mensagens recentes. Texto livre do turno em
    andamento, identificadores e respostas intermediarias ficam de fora.

    Retencao e limpeza: o arquivo e local (sugestao: `var/` do Settings,
    ignorado pelo Git) e nao expira sozinho; remova sessoes encerradas com
    `clear()` e apague o arquivo para purga total. Falhas de leitura de
    dados corrompidos e de acesso ao banco viram `RepositoryError`.
    """

    def __init__(self, path: Path) -> None:
        """Configura o caminho do banco sem realizar acesso imediato."""
        self._path = path

    def save(
        self,
        session_id: str,
        state: ConversationState,
        history: Sequence[BaseMessage],
    ) -> None:
        """Substitui o checkpoint da sessao pelos dados atuais.

        Raises:
            RepositoryError: Se o banco nao puder ser atualizado.
        """
        payload = (
            session_id,
            json.dumps(state.model_dump(mode="json"), ensure_ascii=False),
            json.dumps(messages_to_dict(list(history)), ensure_ascii=False),
            datetime.now(UTC).isoformat(),
        )
        try:
            with closing(sqlite3.connect(self._path)) as connection:
                connection.execute(_CREATE_CHECKPOINTS)
                connection.execute(
                    "INSERT OR REPLACE INTO checkpoints"
                    " (session_id, conversation, history, updated_at)"
                    " VALUES (?, ?, ?, ?)",
                    payload,
                )
                connection.commit()
        except (OSError, sqlite3.Error, ValueError, TypeError) as error:
            raise RepositoryError("session checkpoint could not be saved") from error

    def load(
        self, session_id: str
    ) -> tuple[ConversationState, list[BaseMessage]] | None:
        """Retorna estado e historico validos, ou None quando ausentes.

        Raises:
            RepositoryError: Se o banco falhar ou o checkpoint for invalido.
        """
        try:
            with closing(sqlite3.connect(self._path)) as connection:
                connection.execute(_CREATE_CHECKPOINTS)
                connection.commit()
                row = connection.execute(
                    "SELECT conversation, history FROM checkpoints"
                    " WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
        except (OSError, sqlite3.Error) as error:
            raise RepositoryError("session checkpoint could not be read") from error
        if row is None:
            return None
        try:
            state = ConversationState.model_validate(json.loads(row[0]))
            history = messages_from_dict(json.loads(row[1]))
        except (
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            ValidationError,
        ) as error:
            raise RepositoryError("session checkpoint is invalid") from error
        return state, history

    def clear(self, session_id: str) -> None:
        """Remove o checkpoint da sessao, sem falhar quando ausente.

        Raises:
            RepositoryError: Se o banco nao puder ser atualizado.
        """
        try:
            with closing(sqlite3.connect(self._path)) as connection:
                connection.execute(_CREATE_CHECKPOINTS)
                connection.execute(
                    "DELETE FROM checkpoints WHERE session_id = ?", (session_id,)
                )
                connection.commit()
        except (OSError, sqlite3.Error) as error:
            raise RepositoryError("session checkpoint could not be cleared") from error


def build_graph(dependencies: GraphDependencies) -> ConversationGraph:
    """Compila o grafo de um turno sem checkpointer persistente."""
    builder = StateGraph(GraphState)

    def _turn_context(state: GraphState) -> TurnContext:
        context = state.get("context")
        if context is None:
            context = TurnContext(
                llm=dependencies.llm,
                turn_id=state["turn_id"],
                recent_messages=_previous_messages(state),
            )
        return context

    def triage_node(state: GraphState) -> GraphUpdate:
        if not state["conversation"].authenticated:
            state["conversation"].active_agent = Agent.TRIAGE
        context = _turn_context(state)
        reply = handle_triage(
            state["conversation"],
            state["user_text"],
            dependencies.authentication,
            context=context,
        )
        # Texto que a triagem roteou nao pode ser relido pelo especialista como
        # troca de fluxo; ele so aproveita valor, moeda ou esclarecimento.
        context.text_classified = reply == HANDOFF_REPLY
        update = _handler_update(state, reply, Agent.TRIAGE, context)
        resumed_text = _resumed_request(state["conversation"])
        if resumed_text is not None:
            # O especialista assume o turno agora; sem esta troca ele leria a
            # data de nascimento e perguntaria de novo o que o cliente ja disse.
            update["user_text"] = resumed_text
        return update

    def credit_node(state: GraphState) -> GraphUpdate:
        context = _turn_context(state)
        reply = handle_credit(
            state["conversation"],
            state["user_text"],
            dependencies.credit,
            context=context,
        )
        return _handler_update(state, reply, Agent.CREDIT, context)

    def interview_node(state: GraphState) -> GraphUpdate:
        context = _turn_context(state)
        reply = handle_credit_interview(
            state["conversation"],
            state["user_text"],
            dependencies.credit_interview,
            context=context,
        )
        return _handler_update(state, reply, Agent.CREDIT_INTERVIEW, context)

    def exchange_node(state: GraphState) -> GraphUpdate:
        context = _turn_context(state)
        reply = handle_exchange(
            state["conversation"],
            state["user_text"],
            dependencies.exchange,
            context=context,
        )
        return _handler_update(state, reply, Agent.EXCHANGE, context)

    def knowledge_node(state: GraphState) -> GraphUpdate:
        context = _turn_context(state)
        reply = handle_knowledge(
            state["conversation"],
            state["user_text"],
            dependencies.knowledge,
            context=context,
        )
        return _handler_update(state, reply, Agent.KNOWLEDGE, context)

    def humanize_node(state: GraphState) -> GraphUpdate:
        reply = humanize_reply(
            state["conversation"],
            state["reply"],
            dependencies.llm,
            state["turn_id"],
            responding_agent=state.get("responding_agent"),
            recent_messages=_previous_messages(state),
            user_text=state["user_text"],
        )
        return {"reply": reply}

    builder.add_node("triage", triage_node)
    builder.add_node("credit", credit_node)
    builder.add_node("credit_interview", interview_node)
    builder.add_node("exchange", exchange_node)
    builder.add_node("knowledge", knowledge_node)
    builder.add_node("humanize", humanize_node)
    builder.add_node("limit_guard", _limit_guard)
    builder.add_node("finalize", _finalize)
    builder.add_conditional_edges(START, route_entry)
    builder.add_conditional_edges("triage", route_after_triage)
    builder.add_conditional_edges("credit_interview", route_after_interview)
    builder.add_conditional_edges("credit", route_after_specialist)
    builder.add_conditional_edges("exchange", route_after_specialist)
    builder.add_edge("knowledge", "humanize")
    builder.add_edge("limit_guard", "finalize")
    builder.add_edge("humanize", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile(name="banco-agil-conversation")


def _resumed_request(conversation: ConversationState) -> str | None:
    """Consome o texto do pedido que acabou de ser retomado, se houver.

    Enquanto a autenticacao corre, `deferred_intent` segue preenchido e o texto
    fica guardado. Quando a triagem retoma o pedido ela zera a intencao, e e
    esse par - intencao vazia com texto presente - que marca o turno da entrega.
    """
    if conversation.deferred_intent is not None:
        return None
    texto = conversation.deferred_request
    if texto is None:
        return None
    conversation.deferred_request = None
    return texto


def _handler_update(
    state: GraphState, reply: str, agent: Agent, context: TurnContext
) -> GraphUpdate:
    return {
        "conversation": state["conversation"],
        "reply": reply,
        "step_count": state["step_count"] + 1,
        "responding_agent": agent,
        "context": context,
    }


def _limit_guard(state: GraphState) -> GraphUpdate:
    del state
    return {
        "reply": "Não foi possível continuar esta solicitação. Tente novamente.",
    }


def _finalize(state: GraphState) -> GraphUpdate:
    reply = state["reply"] or "Não foi possível processar a solicitação."
    return {
        "messages": [*state["messages"], AIMessage(content=reply)][-6:],
        "reply": reply,
    }


def _previous_messages(state: GraphState) -> list[AIMessage | HumanMessage]:
    return [
        message
        for message in state["messages"][:-1]
        if isinstance(message, (AIMessage, HumanMessage))
    ][-5:]
