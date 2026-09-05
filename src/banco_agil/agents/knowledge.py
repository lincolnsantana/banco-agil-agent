"""No deterministico que explica as regras do atendimento."""

from banco_agil.agents._shared import (
    HELP_REPLY,
    authentication_reply_if_missing,
    end_reply_if_requested,
    is_help_request,
)
from banco_agil.agents.state import ConversationState
from banco_agil.domain.enums import Agent, Intent
from banco_agil.services.knowledge import KnowledgeService

_UNKNOWN_TOPIC_REPLY = (
    "Sobre isso eu não sei responder. Posso cuidar de limite, aumento, "
    "entrevista de crédito e cotação de moedas. Qual deles ajuda agora?"
)


def handle_knowledge(
    state: ConversationState,
    user_text: str,
    service: KnowledgeService,
) -> str:
    """Responde a duvida a partir do catalogo e devolve a conversa ao fluxo.

    Sem correspondencia no catalogo, admite que nao sabe. Inventar explicacao
    seria pior do que a resposta seca que este no veio corrigir.
    """
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
        state.intent = Intent.UNKNOWN
        state.active_agent = Agent.TRIAGE
        return _UNKNOWN_TOPIC_REPLY

    # A duvida foi respondida: o proximo turno volta a ser roteado do zero,
    # e o servico sugerido no texto fica como intencao pendente de confirmar.
    state.intent = Intent.UNKNOWN
    state.active_agent = Agent.TRIAGE
    if entry.follow_up is not None:
        state.pending_flow = entry.follow_up
    return entry.answer
