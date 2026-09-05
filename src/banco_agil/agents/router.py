"""Rotas condicionais e limites de passos do grafo conversacional."""

from typing import Literal, TypedDict

from langchain_core.messages import BaseMessage

from banco_agil.agents._shared import HANDOFF_REPLY
from banco_agil.agents.state import ConversationState
from banco_agil.domain.enums import Agent

MAX_HANDLER_STEPS = 2
NodeRoute = Literal[
    "triage",
    "credit",
    "credit_interview",
    "exchange",
    "knowledge",
    "limit_guard",
    "humanize",
    "finalize",
]


class GraphState(TypedDict):
    """Estado interno de uma execucao de turno no LangGraph."""

    conversation: ConversationState
    messages: list[BaseMessage]
    user_text: str
    turn_id: str
    reply: str
    responding_agent: Agent | None
    step_count: int


class GraphUpdate(TypedDict, total=False):
    """Atualizacao parcial produzida por um no do grafo."""

    conversation: ConversationState
    messages: list[BaseMessage]
    reply: str
    responding_agent: Agent | None
    step_count: int


def route_entry(state: GraphState) -> NodeRoute:
    """Seleciona o primeiro no impondo encerramento e autenticacao."""
    if state["conversation"].ended:
        return "finalize"
    if state["step_count"] >= MAX_HANDLER_STEPS:
        return "limit_guard"
    if not state["conversation"].authenticated:
        return "triage"
    return _route_for_agent(state["conversation"].active_agent)


def route_after_triage(state: GraphState) -> NodeRoute:
    """Continua no mesmo turno apenas para uma operacao ja identificada."""
    if state["conversation"].ended:
        return "finalize"
    if state["step_count"] >= MAX_HANDLER_STEPS:
        return "limit_guard"
    if state["conversation"].authenticated:
        active_agent = state["conversation"].active_agent
        if active_agent in {
            Agent.CREDIT,
            Agent.CREDIT_INTERVIEW,
            Agent.EXCHANGE,
            Agent.KNOWLEDGE,
        }:
            return _route_for_agent(active_agent)
    return "humanize"


def route_after_specialist(state: GraphState) -> NodeRoute:
    """Entrega o turno a outro especialista quando o cliente mudou de pedido.

    O especialista sinaliza a entrega com a resposta canonica de transicao; o
    destino responde no mesmo turno, e o cliente nunca ve a troca.
    """
    if state["conversation"].ended:
        return "finalize"
    if state["reply"] != HANDOFF_REPLY:
        return "humanize"
    if state["step_count"] >= MAX_HANDLER_STEPS:
        return "limit_guard"
    return _route_for_agent(state["conversation"].active_agent)


def route_after_interview(state: GraphState) -> NodeRoute:
    """Encaminha entrevista concluida para reanalise imediata de credito."""
    if state["conversation"].ended:
        return "finalize"
    if (
        state["conversation"].active_agent is Agent.CREDIT
        and state["conversation"].credit_reanalysis_pending
    ):
        if state["step_count"] >= MAX_HANDLER_STEPS:
            return "limit_guard"
        return "credit"
    return route_after_specialist(state)


def _route_for_agent(agent: Agent) -> NodeRoute:
    if agent is Agent.CREDIT:
        return "credit"
    if agent is Agent.CREDIT_INTERVIEW:
        return "credit_interview"
    if agent is Agent.EXCHANGE:
        return "exchange"
    if agent is Agent.KNOWLEDGE:
        return "knowledge"
    return "triage"
