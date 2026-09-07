"""No que explica as regras do atendimento a partir do catalogo curado."""

from collections.abc import Sequence

from langchain_core.messages import BaseMessage

from banco_agil.agents._shared import (
    HELP_REPLY,
    authentication_reply_if_missing,
    detects_information_question,
    end_reply_if_requested,
    format_money,
    is_help_request,
)
from banco_agil.agents.state import ConversationState
from banco_agil.agents.understanding import TurnContext, resolve_context
from banco_agil.domain.enums import Agent, Intent
from banco_agil.domain.models import Client
from banco_agil.integrations.llm import StructuredLlm
from banco_agil.knowledge.catalog import KnowledgeEntry
from banco_agil.services.knowledge import KnowledgeService

_UNKNOWN_TOPIC_REPLY = (
    "Sobre isso eu não sei responder. Posso cuidar de limite, aumento, "
    "entrevista de crédito e cotação de moedas. Qual deles ajuda agora?"
)
# Entradas em que o numero da propria faixa do cliente torna a regra concreta.
_LIMIT_RULE_TOPICS = frozenset({"why_rejected", "score_defines_limit"})
_SCORE_TOPICS = frozenset({"what_is_score", "how_score_is_calculated"})


def handle_knowledge(
    state: ConversationState,
    user_text: str,
    service: KnowledgeService,
    *,
    context: TurnContext | None = None,
    llm: StructuredLlm | None = None,
    turn_id: str = "",
    recent_messages: Sequence[BaseMessage] = (),
) -> str:
    """Responde a duvida pelo catalogo, fundamentada nos fatos do cliente.

    O texto vem sempre do catalogo; quando os termos nao bastam, o LLM aponta
    o topico, e o Python confere que ele existe. Os fatos anexados (score,
    teto da faixa) vem do estado e do repositorio, nunca do modelo. Sem
    correspondencia, admite que nao sabe em vez de inventar.
    """
    context = resolve_context(context, llm, turn_id, recent_messages)
    end_reply = end_reply_if_requested(state, user_text)
    if end_reply is not None:
        return end_reply
    authentication_reply = authentication_reply_if_missing(state)
    if authentication_reply is not None:
        return authentication_reply

    if is_help_request(user_text):
        return HELP_REPLY

    entry = service.find(user_text)
    if entry is None:
        understanding = context.understand(user_text, Intent.INFORMATION)
        if understanding is not None and understanding.knowledge_topic is not None:
            entry = service.entry_for(understanding.knowledge_topic)

    # A duvida foi tratada: o proximo turno volta a ser roteado do zero.
    state.intent = Intent.UNKNOWN
    state.active_agent = Agent.TRIAGE
    if entry is None:
        return _UNKNOWN_TOPIC_REPLY
    if entry.follow_up is not None:
        state.pending_flow = entry.follow_up
    return _answer_with_facts(entry, state.authenticated_client, service)


_UNKNOWN_MID_STEP_REPLY = "Sobre isso eu não sei responder."


def explanation_for_pending_step(
    user_text: str,
    service: KnowledgeService,
    client: Client | None,
    pending_question: str,
    default_topic: str | None = None,
) -> str | None:
    """Explica a duvida do cliente sem descartar o passo em andamento.

    Pergunta no meio de um fluxo nao e resposta nem desistencia: quem quer
    entender por que a renda e pedida continua querendo a entrevista. A
    explicacao vem do catalogo, com os fatos do cliente quando cabem, e o passo
    e repetido em seguida, de modo que nada do que ja foi coletado se perde.

    `default_topic` e a duvida provavel do passo, usada quando os termos nao
    alcancam nenhuma entrada. E o contexto decidindo o sentido da pergunta:
    "quanto eu posso pedir" no passo do valor e sobre a faixa de score, e a
    mesma pergunta valeria outra explicacao em outro passo. Sem isso, a
    alternativa seria alargar os termos do catalogo, o que rouba as perguntas
    das entradas vizinhas.
    """
    if not detects_information_question(user_text):
        return None
    entry = service.find(user_text)
    if entry is None and default_topic is not None:
        entry = service.entry_for(default_topic)
    if entry is None:
        return f"{_UNKNOWN_MID_STEP_REPLY} {pending_question}"
    return f"{_explanation_only(entry, client, service)} {pending_question}"


def _explanation_only(
    entry: KnowledgeEntry,
    client: Client | None,
    service: KnowledgeService,
) -> str:
    """Devolve a explicacao sem a pergunta final propria do catalogo.

    No meio de um fluxo a pergunta que vale e a do passo pendente; manter as
    duas deixaria o cliente com duas perguntas e nenhuma prioridade.
    """
    answer = _answer_with_facts(entry, client, service)
    body, separator, question = answer.rpartition(". ")
    if separator and question.endswith("?"):
        return f"{body}."
    return answer


def _answer_with_facts(
    entry: KnowledgeEntry, client: Client | None, service: KnowledgeService
) -> str:
    """Insere o fato do cliente entre a explicacao e a pergunta final.

    A redacao pelo LLM exige que os numeros da resposta sejam subconjunto do
    canonico e que ele termine em pergunta, entao o fato entra aqui, no texto
    validado, e nunca na reescrita.
    """
    facts = _facts_for(entry, client, service)
    body, separator, question = entry.answer.rpartition(". ")
    if facts is None or not separator or not question.endswith("?"):
        return entry.answer
    return f"{body}. {facts} {question}"


def _facts_for(
    entry: KnowledgeEntry, client: Client | None, service: KnowledgeService
) -> str | None:
    if client is None:
        return None
    if entry.key in _LIMIT_RULE_TOPICS:
        maximum = service.max_limit_for(client.credit_score)
        if maximum is None:
            return f"Hoje seu score é {client.credit_score}."
        return (
            f"Hoje seu score é {client.credit_score} e sua faixa permite até "
            f"R$ {format_money(maximum)} de limite."
        )
    if entry.key in _SCORE_TOPICS:
        return f"Seu score atual é {client.credit_score}."
    return None
