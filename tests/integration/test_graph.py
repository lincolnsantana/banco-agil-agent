"""Testes de integracao do grafo e do servico de conversa."""

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import TypeVar

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel

from banco_agil.agents.graph import GraphDependencies, build_graph
from banco_agil.agents.router import MAX_HANDLER_STEPS
from banco_agil.agents.state import ConversationState, CreditInterviewDraft
from banco_agil.domain.enums import (
    Agent,
    CreditRequestStatus,
    EmploymentType,
    EndReason,
)
from banco_agil.domain.models import Client, CreditRequest, ExchangeQuote
from banco_agil.services.authentication import AuthenticationService
from banco_agil.services.conversation import ConversationService
from banco_agil.services.credit import CreditService
from banco_agil.services.credit_interview import CreditInterviewService
from banco_agil.services.exchange import ExchangeService

OutputModel = TypeVar("OutputModel", bound=BaseModel)
FIXED_TIME = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)


@dataclass
class MemoryClientRepository:
    """Mantem um cliente ficticio e registra atualizacoes de score."""

    client: Client
    updated_scores: list[int] = field(default_factory=list)

    def find_by_cpf(self, cpf: str) -> Client | None:
        """Retorna o cliente quando o CPF coincide."""
        return self.client if cpf == self.client.cpf else None

    def update_credit_score(self, cpf: str, credit_score: int) -> Client:
        """Atualiza o cliente em memoria."""
        if cpf != self.client.cpf:
            raise ValueError("client not found")
        self.updated_scores.append(credit_score)
        self.client = self.client.model_copy(update={"credit_score": credit_score})
        return self.client


@dataclass
class MemoryScoreLimitRepository:
    """Retorna limite maximo fixo para qualquer score."""

    maximum_limit: Decimal = Decimal("5000.00")
    queried_scores: list[int] = field(default_factory=list)

    def find_max_limit(self, score: int) -> Decimal:
        """Registra o score consultado."""
        self.queried_scores.append(score)
        return self.maximum_limit


@dataclass
class MemoryCreditRequestRepository:
    """Registra criacao e finalizacao na mesma solicitacao."""

    requests: list[CreditRequest] = field(default_factory=list)

    def create(self, request: CreditRequest) -> CreditRequest:
        """Registra uma solicitacao pendente."""
        self.requests.append(request)
        return request

    def finalize(
        self,
        request: CreditRequest,
        status: CreditRequestStatus,
    ) -> CreditRequest:
        """Substitui a mesma solicitacao pelo status final."""
        finalized = request.model_copy(update={"status": status})
        self.requests[-1] = finalized
        return finalized


@dataclass
class RecordingExchangeProvider:
    """Retorna cotacao fixa e registra os pares consultados."""

    calls: list[tuple[str, str]] = field(default_factory=list)

    def get_exchange_rate(
        self,
        base_currency: str,
        quote_currency: str,
    ) -> ExchangeQuote:
        """Retorna cotacao ficticia confirmada."""
        self.calls.append((base_currency, quote_currency))
        return ExchangeQuote(
            base_currency=base_currency,
            quote_currency=quote_currency,
            rate=Decimal("5.25"),
            source="Fonte Fictícia",
            quoted_at=FIXED_TIME,
        )


@dataclass
class RecordingLlm:
    """Registra uma classificacao estruturada sem rede."""

    response: object
    calls: list[list[BaseMessage]] = field(default_factory=list)

    def invoke_structured(
        self,
        turn_id: str,
        messages: list[BaseMessage],
        output_schema: type[OutputModel],
        *,
        prompt_version: str | None = None,
    ) -> OutputModel:
        """Valida a resposta configurada."""
        del turn_id, prompt_version
        self.calls.append(messages)
        return output_schema.model_validate(self.response)


@dataclass
class Harness:
    """Agrupa servico e dependencias observaveis do teste."""

    service: ConversationService
    clients: MemoryClientRepository
    scores: MemoryScoreLimitRepository
    requests: MemoryCreditRequestRepository
    exchange: RecordingExchangeProvider


@pytest.fixture
def client() -> Client:
    """Cria cliente ficticio."""
    return Client(
        cpf="01234567890",
        birth_date=date(1990, 5, 20),
        credit_limit=Decimal("2500.00"),
        credit_score=700,
    )


def build_harness(client: Client, llm: RecordingLlm | None = None) -> Harness:
    """Monta grafo com todas as integracoes substituidas por memoria."""
    clients = MemoryClientRepository(client)
    scores = MemoryScoreLimitRepository()
    requests = MemoryCreditRequestRepository()
    exchange = RecordingExchangeProvider()
    dependencies = GraphDependencies(
        authentication=AuthenticationService(clients),
        credit=CreditService(scores, requests, clock=lambda: FIXED_TIME),
        credit_interview=CreditInterviewService(clients),
        exchange=ExchangeService(exchange),
        llm=llm,
    )
    return Harness(
        service=ConversationService(build_graph(dependencies), lambda: "turn-id"),
        clients=clients,
        scores=scores,
        requests=requests,
        exchange=exchange,
    )


def test_graph_forces_unauthenticated_specialist_back_to_triage(
    client: Client,
) -> None:
    harness = build_harness(client)
    state = ConversationState(active_agent=Agent.EXCHANGE)

    turn = harness.service.handle_turn(state, (), "USD-BRL")

    assert state.active_agent is Agent.TRIAGE
    assert "cpf" in turn.reply.casefold()
    assert harness.exchange.calls == []


def test_triage_routes_to_credit_and_returns_final_reply_in_same_turn(
    client: Client,
) -> None:
    harness = build_harness(client)
    state = ConversationState(authenticated_client=client)

    turn = harness.service.handle_turn(state, (), "qual é meu limite?")

    assert "2.500,00" in turn.reply
    assert "prosseguir" not in turn.reply.casefold()
    assert len(turn.history) == 2
    assert isinstance(turn.history[0], HumanMessage)
    assert isinstance(turn.history[1], AIMessage)


def test_triage_routes_to_exchange_in_same_turn(client: Client) -> None:
    harness = build_harness(client)
    state = ConversationState(authenticated_client=client)

    turn = harness.service.handle_turn(state, (), "cotação do dólar")

    assert harness.exchange.calls == [("USD", "BRL")]
    assert "USD-BRL" in turn.reply
    assert "5,25" in turn.reply


def test_interview_completion_reanalyzes_credit_in_same_turn(client: Client) -> None:
    harness = build_harness(client)
    state = ConversationState(
        authenticated_client=client,
        active_agent=Agent.CREDIT_INTERVIEW,
        requested_limit=Decimal("4000.00"),
        interview_draft=CreditInterviewDraft(
            consent_given=True,
            monthly_income=Decimal("5000.00"),
            employment_type=EmploymentType.FORMAL,
            monthly_expenses=Decimal("2000.00"),
            dependents=1,
        ),
    )

    turn = harness.service.handle_turn(state, (), "não")

    assert "aprovada" in turn.reply.casefold()
    assert harness.clients.updated_scores
    assert harness.requests.requests[0].requested_limit == Decimal("4000.00")
    assert harness.requests.requests[0].status is CreditRequestStatus.APPROVED
    assert not state.credit_reanalysis_pending


@pytest.mark.parametrize("agent", list(Agent))
def test_end_request_preempts_every_graph_node(client: Client, agent: Agent) -> None:
    llm = RecordingLlm({"intent": "credit_limit"})
    harness = build_harness(client, llm)
    state = ConversationState(authenticated_client=client, active_agent=agent)

    turn = harness.service.handle_turn(state, (), "encerrar")

    assert state.ended
    assert state.end_reason is EndReason.USER_REQUEST
    assert "encerrado" in turn.reply.casefold()
    assert harness.exchange.calls == []
    assert harness.requests.requests == []
    assert llm.calls == []


def test_graph_step_guard_returns_controlled_reply(client: Client) -> None:
    harness = build_harness(client)
    graph = harness.service.graph

    result = graph.invoke(
        {
            "conversation": ConversationState(authenticated_client=client),
            "messages": [HumanMessage(content="limite")],
            "user_text": "limite",
            "turn_id": "guard-turn",
            "reply": "",
            "step_count": MAX_HANDLER_STEPS,
        }
    )

    assert "continuar" in str(result["reply"]).casefold()
    assert harness.requests.requests == []
    assert harness.exchange.calls == []


def test_ambiguous_intent_uses_one_llm_call_and_stays_in_triage(
    client: Client,
) -> None:
    llm = RecordingLlm({"intent": "other"})
    harness = build_harness(client, llm)
    state = ConversationState(authenticated_client=client)

    turn = harness.service.handle_turn(state, (), "preciso resolver outra coisa")

    assert len(llm.calls) == 1
    assert state.active_agent is Agent.TRIAGE
    assert "limite" in turn.reply.casefold()
    assert "cotação" in turn.reply.casefold()


def test_llm_receives_at_most_six_sanitized_conversation_messages(
    client: Client,
) -> None:
    llm = RecordingLlm({"intent": "other"})
    harness = build_harness(client, llm)
    state = ConversationState(authenticated_client=client)
    history: tuple[BaseMessage, ...] = (
        HumanMessage(content="Meu CPF é 01234567890"),
        AIMessage(content="Como posso ajudar?"),
        HumanMessage(content="preciso de ajuda"),
        AIMessage(content="Qual serviço deseja?"),
        HumanMessage(content="outra coisa"),
        AIMessage(content="Pode explicar melhor?"),
    )

    harness.service.handle_turn(state, history, "preciso resolver outra coisa")

    sent_messages = llm.calls[0]
    assert len(sent_messages[1:]) <= 6
    assert "01234567890" not in str(sent_messages)


def test_history_keeps_only_six_recent_messages(client: Client) -> None:
    harness = build_harness(client)
    state = ConversationState()
    history: tuple[BaseMessage, ...] = ()

    for index in range(4):
        turn = harness.service.handle_turn(state, history, f"mensagem {index}")
        history = turn.history

    assert len(history) == 6
    assert [message.content for message in history] == [
        "mensagem 1",
        "Olá! Para começar, informe seu CPF com 11 dígitos.",
        "mensagem 2",
        "Olá! Para começar, informe seu CPF com 11 dígitos.",
        "mensagem 3",
        "Olá! Para começar, informe seu CPF com 11 dígitos.",
    ]


def test_conversation_rejects_empty_message(client: Client) -> None:
    harness = build_harness(client)

    with pytest.raises(ValueError, match="message cannot be empty"):
        harness.service.handle_turn(ConversationState(), (), "   ")
