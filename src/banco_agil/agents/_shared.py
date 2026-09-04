"""Comportamentos pequenos compartilhados pelos especialistas."""

import re
import unicodedata
from collections.abc import Sequence
from decimal import Decimal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from banco_agil.agents.state import ConversationState
from banco_agil.domain.enums import Agent, EndReason
from banco_agil.domain.exceptions import IntegrationError
from banco_agil.domain.models import EndServiceResult
from banco_agil.integrations.llm import StructuredLlm
from banco_agil.prompts.renderer import render_prompt
from banco_agil.tools.banking import end_service

_END_REQUESTS = {
    "cancelar atendimento",
    "encerrar",
    "encerrar atendimento",
    "fim",
    "parar",
    "quero encerrar",
    "nao quero continuar",
    "sair",
}
_END_PATTERN = re.compile(
    r"^(?:por favor,? )?(?:(?:eu )?(?:quero|gostaria de|pode) )?"
    r"(?:encerrar|encerre|finalizar|finalize|sair)(?: o atendimento)?$"
)
_SAFE_LLM_WORDS = frozenset(
    {
        "ajuda",
        "ajustar",
        "alterar",
        "banco",
        "aumentar",
        "aumento",
        "cambio",
        "consultar",
        "cotacao",
        "dolar",
        "euro",
        "entrevista",
        "limite",
        "modificar",
        "mudar",
        "quero",
        "saber",
        "taxa",
        "cartao",
        "coisa",
        "credito",
        "emprestimo",
        "exterior",
        "financiamento",
        "moeda",
        "outra",
        "pontuacao",
        "preciso",
        "resolver",
        "rever",
        "score",
        "servico",
        "viagem",
    }
)


_FACT_PATTERN = re.compile(
    r"\b[A-Z]{3}-[A-Z]{3}\b|\b\d{4}-\d{2}-\d{2}\b|(?:R\$\s*)?(?:\d[\d.,]*\d|\d)"
)
_FACT_TOKEN_PATTERN = re.compile(r"\[DADO_\d+\]")
_LEAK_PHRASES = (
    "system prompt",
    "prompt do sistema",
    "instrucao de sistema",
    "como modelo de linguagem",
    "minhas instrucoes",
    "regras internas",
)
_MAX_REWRITE_LENGTH = 600


class RewrittenReply(BaseModel):
    """Reescrita integral da resposta canônica, com fatos mascarados."""

    reply: str = Field(min_length=1, max_length=_MAX_REWRITE_LENGTH)


def normalized_text(value: str) -> str:
    """Normaliza texto curto para parsers deterministicos."""
    without_accents = "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold().strip())
        if not unicodedata.combining(character)
    )
    return " ".join(without_accents.split())


def end_reply_if_requested(state: ConversationState, user_text: str) -> str | None:
    """Encerra o atendimento antes de qualquer outra operacao."""
    normalized = normalized_text(user_text)
    if normalized not in _END_REQUESTS and _END_PATTERN.search(normalized) is None:
        return None
    end_conversation(state, EndReason.USER_REQUEST)
    return "Atendimento encerrado. Quando precisar, estaremos à disposição."


def end_conversation(state: ConversationState, reason: EndReason) -> None:
    """Executa a tool de encerramento e aplica seu resultado ao estado."""
    result = EndServiceResult.model_validate(end_service.invoke({"reason": reason}))
    state.end(result.reason)


def authentication_reply_if_missing(state: ConversationState) -> str | None:
    """Redireciona operacao protegida para coleta de credenciais."""
    if state.authenticated:
        return None
    state.active_agent = Agent.TRIAGE
    return "Para continuar, preciso confirmar seus dados. Por favor, informe seu CPF."


def parse_confirmation(value: str) -> bool | None:
    """Converte respostas curtas de consentimento sem usar LLM."""
    normalized = normalized_text(value)
    if normalized in {"sim", "aceito", "concordo", "pode", "quero"}:
        return True
    if normalized in {"nao", "recuso", "prefiro nao"}:
        return False
    return None


def sanitize_user_text(value: str) -> str:
    """Seleciona somente termos nao sensiveis antes de chamar o LLM."""
    words = re.findall(r"[a-z]+", normalized_text(value))
    return " ".join(word for word in words if word in _SAFE_LLM_WORDS)


def humanize_reply(
    state: ConversationState,
    canonical_reply: str,
    llm: StructuredLlm | None,
    turn_id: str,
    *,
    responding_agent: Agent | None,
    recent_messages: Sequence[BaseMessage],
    user_text: str,
) -> str:
    """Reescreve a resposta canônica sem permitir mudança nos fatos.

    O LLM recebe o canônico com fatos mascarados e pode redigir a resposta
    final completa. Falhas, saídas inseguras e orçamento consumido retornam
    silenciosamente ao texto determinístico.
    """
    if (
        llm is None
        or not turn_id
        or state.ended
        or responding_agent is None
        or responding_agent is Agent.TRIAGE
        or llm.calls_remaining(turn_id) == 0
    ):
        return canonical_reply

    masked_reply, facts = _mask_facts(canonical_reply)
    prompt_state = state.model_copy(deep=True)
    prompt_state.active_agent = responding_agent
    rendered = render_prompt(prompt_state)
    safe_user_text = sanitize_user_text(user_text) or "pedido bancario"
    messages: list[BaseMessage] = [
        rendered.system_message,
        *_safe_history(recent_messages),
        HumanMessage(
            content=(
                "Redija a resposta final completa a partir do texto validado "
                "abaixo. Preserve cada marcador [DADO_N] exatamente como está, "
                "sem criar fatos, números, decisões ou perguntas novos. "
                f"Texto validado: {masked_reply} "
                f"Contexto seguro: {safe_user_text}."
            )
        ),
    ]
    try:
        result = llm.invoke_structured(
            turn_id,
            messages,
            RewrittenReply,
            prompt_version=rendered.prompt_version,
        )
    except IntegrationError:
        return canonical_reply

    rewritten = " ".join(result.reply.split())
    restored = _restore_facts(rewritten, facts)
    if restored is None or not _preserves_decision(restored, canonical_reply):
        return canonical_reply
    return restored


def _mask_facts(text: str) -> tuple[str, dict[str, str]]:
    """Substitui pares, datas e números por marcadores opacos."""
    facts: dict[str, str] = {}

    def _replace(match: re.Match[str]) -> str:
        token = f"[DADO_{len(facts) + 1}]"
        facts[token] = match.group()
        return token

    return _FACT_PATTERN.sub(_replace, text), facts


def _restore_facts(masked_reply: str, facts: dict[str, str]) -> str | None:
    """Recoloca os fatos se todos os marcadores forem preservados."""
    if set(_FACT_TOKEN_PATTERN.findall(masked_reply)) != set(facts):
        return None
    restored = masked_reply
    for token, value in facts.items():
        restored = restored.replace(token, value)
    if _FACT_TOKEN_PATTERN.search(restored) is not None:
        return None
    return restored


def _preserves_decision(rewritten: str, canonical_reply: str) -> bool:
    """Exige subconjunto de números, perguntas e ausência de vazamento."""
    normalized_rewritten = normalized_text(rewritten)
    if any(leak in normalized_rewritten for leak in _LEAK_PHRASES):
        return False
    if rewritten.strip().endswith("?") != canonical_reply.strip().endswith("?"):
        return False
    canonical_digits = re.findall(r"\d+", canonical_reply)
    rewritten_digits = re.findall(r"\d+", rewritten)
    remaining = list(canonical_digits)
    for digits in rewritten_digits:
        if digits not in remaining:
            return False
        remaining.remove(digits)
    return True


def _safe_history(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    safe_messages: list[BaseMessage] = []
    for message in messages[-5:]:
        safe_content = sanitize_user_text(str(message.content))
        if not safe_content:
            continue
        message_type = AIMessage if isinstance(message, AIMessage) else HumanMessage
        safe_messages.append(message_type(content=safe_content))
    return safe_messages


def format_money(value: Decimal) -> str:
    """Formata valor decimal para exibicao bancaria em pt-BR."""
    formatted = f"{value:,.2f}"
    return formatted.replace(",", "_").replace(".", ",").replace("_", ".")
