"""Testes da rota informativa e do catalogo explicativo."""

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import TypeVar

import pytest
from pydantic import BaseModel

from banco_agil.agents._shared import detects_information_question
from banco_agil.agents.knowledge import handle_knowledge
from banco_agil.agents.state import ConversationState
from banco_agil.domain.enums import Agent, Intent
from banco_agil.domain.models import Client
from banco_agil.knowledge.catalog import (
    KNOWLEDGE_CATALOG,
    MAX_ANSWER_LENGTH,
    KnowledgeEntry,
)
from banco_agil.services.knowledge import KnowledgeRetriever, KnowledgeService

_Model = TypeVar("_Model", bound=BaseModel)


@pytest.fixture
def client() -> Client:
    """Cria cliente ficticio autenticado."""
    return Client(
        cpf="01234567890",
        birth_date=date(1990, 5, 20),
        credit_limit=Decimal("2500.00"),
        credit_score=700,
    )


# As oito perguntas medidas antes da mudanca, quando 8 de 8 disparavam a acao
# errada em vez de responder.
@pytest.mark.parametrize(
    ("pergunta", "chave"),
    (
        ("por que meu aumento foi rejeitado?", "why_rejected"),
        ("o que é score de crédito?", "what_is_score"),
        ("como vocês calculam meu score?", "how_score_is_calculated"),
        ("quanto tempo demora a análise?", "analysis_time"),
        ("posso pedir aumento de novo depois?", "can_ask_again"),
        ("meu score de 750 é bom?", "what_is_score"),
        ("o que acontece se eu não pagar a fatura?", "out_of_scope_billing"),
        ("vocês cobram taxa para aumentar o limite?", "increase_fee"),
    ),
)
def test_measured_questions_reach_the_right_explanation(
    pergunta: str,
    chave: str,
) -> None:
    entry = KnowledgeService().find(pergunta)

    assert entry is not None
    assert entry.key == chave


@pytest.mark.parametrize(
    "pergunta",
    (
        "por que meu aumento foi rejeitado?",
        "o que é score de crédito?",
        "quanto tempo demora a análise?",
        "vocês cobram taxa para aumentar o limite?",
        "de onde vem a cotação?",
        "por que preciso informar meu cpf?",
    ),
)
def test_questions_are_recognized_as_questions(pergunta: str) -> None:
    assert detects_information_question(pergunta) is True


@pytest.mark.parametrize(
    "pedido",
    (
        "quero aumentar meu limite",
        "preciso de mais limite",
        "quero um limite maior",
        "quero fazer a entrevista de crédito",
        "qual é o meu limite?",
        "quero aumentar meu score",
        "solicitar aumento de crédito",
        "quero saber meu limite antes de pedir aumento",
        "quero ver a cotação do dólar",
    ),
)
def test_commands_are_never_read_as_questions(pedido: str) -> None:
    # A rota informativa nao pode engolir pedido de operacao.
    assert detects_information_question(pedido) is False


def test_catalog_answers_survive_the_humanization_guards() -> None:
    """Trava os invariantes ditados por `_preserves_decision`.

    Resposta longa estoura o teto da reescrita; digito novo derruba a checagem
    de numeros; fim sem interrogacao quebra a paridade. Em qualquer um deles a
    reescrita e descartada e o cliente ve o texto seco de volta.
    """
    for entry in KNOWLEDGE_CATALOG:
        assert len(entry.answer) <= MAX_ANSWER_LENGTH, entry.key
        assert re.search(r"\d", entry.answer) is None, entry.key
        assert entry.answer.rstrip().endswith("?"), entry.key


def test_catalog_keys_are_unique() -> None:
    chaves = [entry.key for entry in KNOWLEDGE_CATALOG]

    assert len(set(chaves)) == len(chaves)


def test_unmatched_question_admits_ignorance_instead_of_guessing(
    client: Client,
) -> None:
    state = ConversationState(
        authenticated_client=client,
        active_agent=Agent.KNOWLEDGE,
        intent=Intent.INFORMATION,
    )
    vazio = KnowledgeService(catalog=())

    reply = handle_knowledge(state, "qual a cor do seu cartão?", vazio)

    assert "não sei responder" in reply
    assert state.active_agent is Agent.TRIAGE


def test_answer_offers_the_service_the_entry_points_to(client: Client) -> None:
    state = ConversationState(
        authenticated_client=client,
        active_agent=Agent.KNOWLEDGE,
        intent=Intent.INFORMATION,
    )

    handle_knowledge(state, "por que meu aumento foi rejeitado?", KnowledgeService())

    assert state.pending_flow is Intent.CREDIT_INTERVIEW


def test_knowledge_requires_authentication() -> None:
    state = ConversationState(active_agent=Agent.KNOWLEDGE)

    reply = handle_knowledge(state, "o que é score?", KnowledgeService())

    assert "confirmar seus dados" in reply.casefold()
    assert state.active_agent is Agent.TRIAGE


def test_retriever_exposes_the_catalog_through_langchain() -> None:
    documentos = KnowledgeRetriever().invoke("por que meu aumento foi rejeitado?")

    assert documentos
    assert documentos[0].metadata["key"] == "why_rejected"
    assert documentos[0].page_content


def test_ranking_ties_follow_catalog_order_not_key_order() -> None:
    # A ordem e autoral: a explicacao geral vem antes da especifica.
    catalogo = (
        KnowledgeEntry(key="zz_geral", terms=frozenset({"score"}), answer="Geral?"),
        KnowledgeEntry(key="aa_especifico", terms=frozenset({"score"}), answer="Esp?"),
    )

    entry = KnowledgeService(catalog=catalogo).find("meu score")

    assert entry is not None
    assert entry.key == "zz_geral"


def test_second_question_is_not_swallowed_by_a_pending_offer(client: Client) -> None:
    """Regressao de sequencia: uma pergunta seguida de outra.

    A resposta deixa uma oferta pendente. Sem esta guarda, a pergunta seguinte
    caia no confirmador de sim/nao e virava "Nao consegui confirmar".
    """
    from banco_agil.agents.triage import handle_triage

    state = ConversationState(
        authenticated_client=client, pending_flow=Intent.CREDIT_INTERVIEW
    )

    reply = handle_triage(state, "vocês cobram taxa?", _FakeAuth())

    assert "confirmar" not in reply.casefold()
    assert state.active_agent is Agent.KNOWLEDGE


def test_command_supersedes_a_pending_offer(client: Client) -> None:
    from banco_agil.agents.triage import handle_triage

    state = ConversationState(
        authenticated_client=client, pending_flow=Intent.CREDIT_INTERVIEW
    )

    handle_triage(state, "quero aumentar meu limite", _FakeAuth())

    # A oferta anterior caduca: o pedido novo manda.
    assert state.pending_flow is None
    assert state.intent is Intent.LIMIT_INCREASE


def test_vague_answer_still_repeats_the_pending_question(client: Client) -> None:
    from banco_agil.agents.triage import handle_triage

    state = ConversationState(
        authenticated_client=client, pending_flow=Intent.CREDIT_INTERVIEW
    )

    reply = handle_triage(state, "não sei", _FakeAuth())

    # Sem assunto novo, repetir a pergunta continua sendo o certo.
    assert "não consegui confirmar" in reply.casefold()
    assert state.pending_flow is Intent.CREDIT_INTERVIEW


class _FakeAuth:
    """Autenticacao que falha se for chamada com o cliente ja autenticado."""

    def validate_cpf(self, state: ConversationState, cpf: str) -> object:
        raise AssertionError("não deveria revalidar CPF")

    def authenticate(
        self, state: ConversationState, cpf: str, birth_date: str
    ) -> object:
        raise AssertionError("não deveria reautenticar")


@dataclass
class _RecordingLlm:
    response: object
    calls: int = 0

    def calls_remaining(self, turn_id: str) -> int:
        del turn_id
        return 2 - self.calls

    def invoke_structured(
        self,
        turn_id: str,
        messages: object,
        output_schema: type[_Model],
        *,
        prompt_version: str | None = None,
    ) -> _Model:
        del turn_id, messages, prompt_version
        self.calls += 1
        return output_schema.model_validate(self.response)


@dataclass
class _FixedScoreLimits:
    maximum: Decimal

    def find_max_limit(self, score: int) -> Decimal:
        del score
        return self.maximum


def test_knowledge_uses_understood_topic_when_terms_do_not_match(
    client: Client,
) -> None:
    state = ConversationState(authenticated_client=client, active_agent=Agent.KNOWLEDGE)
    llm = _RecordingLlm({"intent": "information", "knowledge_topic": "why_rejected"})

    reply = handle_knowledge(
        state,
        "não entendi o motivo daquela resposta negativa",
        KnowledgeService(),
        llm=llm,
        turn_id="t",
    )

    assert "teto da sua faixa" in reply
    assert state.pending_flow is Intent.CREDIT_INTERVIEW
    assert llm.calls == 1


def test_knowledge_grounds_the_rule_in_the_client_band(client: Client) -> None:
    state = ConversationState(authenticated_client=client, active_agent=Agent.KNOWLEDGE)
    service = KnowledgeService(score_limits=_FixedScoreLimits(Decimal("5000.00")))

    reply = handle_knowledge(state, "por que meu aumento foi rejeitado?", service)

    assert "Hoje seu score é 700 e sua faixa permite até R$ 5.000,00" in reply
    assert reply.endswith("Quer tentar?")


def test_knowledge_without_band_repository_only_states_the_score(
    client: Client,
) -> None:
    state = ConversationState(authenticated_client=client, active_agent=Agent.KNOWLEDGE)

    reply = handle_knowledge(state, "o que é score?", KnowledgeService())

    assert "Seu score atual é 700." in reply
    assert reply.endswith("?")


def test_knowledge_admits_ignorance_when_topic_is_not_in_catalog(
    client: Client,
) -> None:
    state = ConversationState(authenticated_client=client, active_agent=Agent.KNOWLEDGE)
    llm = _RecordingLlm({"intent": "information", "knowledge_topic": "invented"})

    reply = handle_knowledge(
        state, "qual a cor do seu cartão?", KnowledgeService(), llm=llm, turn_id="t"
    )

    assert "não sei responder" in reply
