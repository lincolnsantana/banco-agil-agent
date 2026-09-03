"""Montagem do LangGraph com dependencias bancarias injetadas."""

from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from banco_agil.agents.credit import handle_credit
from banco_agil.agents.credit_interview import handle_credit_interview
from banco_agil.agents.exchange import handle_exchange
from banco_agil.agents.router import (
    GraphState,
    GraphUpdate,
    route_after_interview,
    route_after_triage,
    route_entry,
)
from banco_agil.agents.triage import handle_triage
from banco_agil.domain.enums import Agent
from banco_agil.integrations.llm import StructuredLlm
from banco_agil.services.authentication import AuthenticationService
from banco_agil.services.credit import CreditService
from banco_agil.services.credit_interview import CreditInterviewService
from banco_agil.services.exchange import ExchangeService

ConversationGraph = CompiledStateGraph[GraphState, None, GraphState, GraphState]


@dataclass(frozen=True)
class GraphDependencies:
    """Servicos permitidos aos quatro nos especialistas."""

    authentication: AuthenticationService
    credit: CreditService
    credit_interview: CreditInterviewService
    exchange: ExchangeService
    llm: StructuredLlm | None = None


def build_graph(dependencies: GraphDependencies) -> ConversationGraph:
    """Compila o grafo de um turno sem checkpointer persistente."""
    builder = StateGraph(GraphState)

    def triage_node(state: GraphState) -> GraphUpdate:
        if not state["conversation"].authenticated:
            state["conversation"].active_agent = Agent.TRIAGE
        reply = handle_triage(
            state["conversation"],
            state["user_text"],
            dependencies.authentication,
            llm=dependencies.llm,
            turn_id=state["turn_id"],
            recent_messages=_previous_messages(state),
        )
        return _handler_update(state, reply)

    def credit_node(state: GraphState) -> GraphUpdate:
        reply = handle_credit(
            state["conversation"],
            state["user_text"],
            dependencies.credit,
        )
        return _handler_update(state, reply)

    def interview_node(state: GraphState) -> GraphUpdate:
        reply = handle_credit_interview(
            state["conversation"],
            state["user_text"],
            dependencies.credit_interview,
        )
        return _handler_update(state, reply)

    def exchange_node(state: GraphState) -> GraphUpdate:
        reply = handle_exchange(
            state["conversation"],
            state["user_text"],
            dependencies.exchange,
        )
        return _handler_update(state, reply)

    builder.add_node("triage", triage_node)
    builder.add_node("credit", credit_node)
    builder.add_node("credit_interview", interview_node)
    builder.add_node("exchange", exchange_node)
    builder.add_node("limit_guard", _limit_guard)
    builder.add_node("finalize", _finalize)
    builder.add_conditional_edges(START, route_entry)
    builder.add_conditional_edges("triage", route_after_triage)
    builder.add_conditional_edges("credit_interview", route_after_interview)
    builder.add_edge("credit", "finalize")
    builder.add_edge("exchange", "finalize")
    builder.add_edge("limit_guard", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile(name="banco-agil-conversation")


def _handler_update(state: GraphState, reply: str) -> GraphUpdate:
    return {
        "conversation": state["conversation"],
        "reply": reply,
        "step_count": state["step_count"] + 1,
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
