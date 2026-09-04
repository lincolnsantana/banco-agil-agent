"""Comportamentos pequenos compartilhados pelos especialistas."""

import re
import unicodedata
from collections.abc import Sequence
from decimal import Decimal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from banco_agil.agents.state import ConversationState
from banco_agil.domain.enums import Agent, EndReason, Intent
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
    r"^(?:por favor,?\s*)?"
    r"(?:(?:eu\s+)?(?:quero|desejo|gostaria\s+de|preciso|prefiro)\s+|"
    r"(?:pode|podemos|vamos)\s+)?"
    r"(?:encerrar|encerre|finalizar|finalize|terminar|termine|fechar|feche|parar|sair)"
    r"(?:\s+(?:o|a|este|esta|esse|essa|minha|meu|do|da))?"
    r"(?:\s+(?:atendimento|conversa|chat|sessao|servico))?"
    r"(?:\s+(?:agora|por\s+favor|por\s+aqui))?$"
)
_END_CONTINUATION_PATTERN = re.compile(
    r"^(?:eu\s+)?nao\s+(?:quero|desejo|gostaria\s+de|pretendo)\s+"
    r"(?:mais\s+)?continuar(?:\s+com)?(?:\s+(?:o|a|este|esta|minha|meu))?"
    r"(?:\s+(?:atendimento|conversa|chat|sessao|servico))?$"
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
        "servicos",
        "viagem",
        "fazer",
        "funciona",
        "menu",
        "opcoes",
    }
)

HELP_REPLY = (
    "Claro! Posso consultar seu limite de crédito, solicitar um aumento de "
    "limite, conduzir a entrevista de crédito para revisar seu score e "
    "consultar cotações de moedas como dólar e euro. Por onde quer começar?"
)

_HELP_PHRASES = (
    "o que voce pode fazer",
    "o que voce pode realizar",
    "o que voce faz",
    "o que voce oferece",
    "o que voce realiza",
    "o que pode realizar",
    "o que sabe fazer",
    "quais servicos",
    "que servicos",
    "quais opcoes",
    "suas funcionalidades",
    "suas funcoes",
    "como funciona",
    "me fale sobre os servicos",
    "o que posso fazer",
)


def is_help_request(user_text: str) -> bool:
    """Detecta pergunta sobre o atendimento sem usar LLM."""
    normalized = normalized_text(user_text)
    if normalized in {"ajuda", "menu"}:
        return True
    return any(phrase in normalized for phrase in _HELP_PHRASES)


_HOWTO_MARKERS = (
    "como faco",
    "como fazer",
    "como posso",
    "como consigo",
    "como realizo",
    "como solicitar",
    "como solicito",
    "como pedir",
    "como peco",
    "como consultar",
    "como consulto",
    "como aumento",
    "como aumentar",
    "como melhorar",
    "como ver",
    "como vejo",
    "como funciona",
    "como comeco",
    "como iniciar",
    "me explica",
    "me explique",
    "passo a passo",
    "quais sao os passos",
    "o que preciso para",
    "posso aumentar",
    "e possivel aumentar",
    "tem como aumentar",
    "por que nao consigo aumentar",
    "porque nao consigo aumentar",
)

_FLOW_AFFIRMATIVE_WORDS = frozenset(
    {
        "sim",
        "quero",
        "vamos",
        "claro",
        "gostaria",
        "aceito",
        "concordo",
        "bora",
        "confirmo",
        "confirmado",
        "fechado",
        "prossiga",
        "continue",
        "pode",
        "faca",
        "realize",
        "manda",
        "interesse",
        "topo",
        "ok",
        "okay",
        "beleza",
        "certo",
    }
)
_FLOW_REFUSAL_WORDS = frozenset(
    {
        "quero",
        "vou",
        "prefiro",
        "obrigado",
        "obrigada",
        "valeu",
        "dispenso",
        "recuso",
        "cancela",
        "cancelar",
        "deixa",
        "interesse",
        "pode",
        "precisa",
    }
)
_FLOW_NEVER_WORDS = frozenset({"nunca", "jamais"})


def detect_howto_topic(user_text: str) -> Intent | None:
    """Mapeia pergunta de como-fazer ao fluxo, sem usar LLM."""
    normalized = normalized_text(user_text)
    if not any(marker in normalized for marker in _HOWTO_MARKERS):
        return None
    if any(word in normalized for word in ("entrevista", "score", "pontuacao")):
        return Intent.CREDIT_INTERVIEW
    if any(
        word in normalized
        for word in ("cambio", "cotacao", "dolar", "euro", "libra", "moeda")
    ):
        return Intent.EXCHANGE_RATE
    if "aument" in normalized or "novo limite" in normalized:
        return Intent.LIMIT_INCREASE
    if "limite" in normalized:
        return Intent.CREDIT_LIMIT
    return None


_FLOW_AFFIRMATIVE_PHRASES = frozenset(
    {
        "sim",
        "quero",
        "vamos",
        "claro",
        "com certeza",
        "pode ser",
        "gostaria",
        "aceito",
        "concordo",
        "bora",
        "confirmo",
        "confirmado",
        "fechado",
        "prossiga",
        "continue",
        "pode",
        "pode prosseguir",
        "pode continuar",
        "pode fazer",
        "faca isso",
        "vamos fazer",
        "vamos nessa",
        "tenho interesse",
        "quero realizar",
        "quero fazer",
        "manda ver",
        "manda bala",
        "eu topo",
        "ok",
        "okay",
        "beleza",
        "certo",
        "isso",
        "isso mesmo",
    }
)
_FLOW_NEGATIVE_PHRASES = frozenset(
    {
        "nao",
        "agora nao",
        "depois",
        "prefiro nao",
        "dispenso",
        "cancela",
        "cancelar",
        "melhor nao",
        "deixa",
        "deixa pra la",
        "deixa para depois",
        "mais tarde",
        "outro momento",
        "nao agora",
        "nao tenho interesse",
        "nao precisa",
    }
)


def parse_flow_answer(user_text: str) -> bool | None:
    """Converte resposta ampla à confirmação de fluxo, sem usar LLM."""
    normalized = normalize_short_answer(user_text)
    if normalized in _FLOW_NEGATIVE_PHRASES:
        return False
    if normalized in _FLOW_AFFIRMATIVE_PHRASES:
        return True
    words = set(normalized.split())
    if "nao" in words and words & _FLOW_REFUSAL_WORDS:
        return False
    if words & _FLOW_AFFIRMATIVE_WORDS:
        return True
    if words & _FLOW_NEVER_WORDS:
        return False
    return None


_FACT_PATTERN = re.compile(
    r"\b[A-Z]{3}-[A-Z]{3}\b|\b\d{4}-\d{2}-\d{2}\b|(?:R\$\s*)?(?:\d[\d.,]*\d|\d)"
)
_FACT_TOKEN_PATTERN = re.compile(r"\[DADO_\d+\]")
_USER_TEXT_LIMIT = 280
_MASKED_NUMBER = "esse valor"
_CONTROL_PATTERN = re.compile(r"[\x00-\x1f\x7f]+")
# Colchetes sairiam caros: o cliente poderia forjar um marcador [DADO_N].
_STRUCTURE_PATTERN = re.compile(r"[`<>{}\[\]|]+")
_CPF_TEXT_PATTERN = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
_DATE_TEXT_PATTERN = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b")
_NUMBER_TEXT_PATTERN = re.compile(r"(?:R\$\s*)?\d[\d.,]*")
_LEAK_PHRASES = (
    "system prompt",
    "prompt do sistema",
    "instrucao de sistema",
    "como modelo de linguagem",
    "minhas instrucoes",
    "regras internas",
    "contexto seguro",
    "pergunta do cliente",
    "texto validado",
    "dado_n",
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
    normalized = normalized_text(user_text).strip(".,!?;:'\"()[]-").strip()
    if (
        normalized not in _END_REQUESTS
        and _END_PATTERN.fullmatch(normalized) is None
        and _END_CONTINUATION_PATTERN.fullmatch(normalized) is None
    ):
        return None
    end_conversation(state, EndReason.USER_REQUEST)
    return (
        "Atendimento encerrado. Quando precisar de um novo atendimento, "
        "basta informar seu CPF com 11 dígitos."
    )


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


_SHORT_ANSWER_PUNCTUATION = ".,!?;:'\"()[]-"


def normalize_short_answer(value: str) -> str:
    """Remove acentos, caixa e pontuação lateral de respostas curtas."""
    return normalized_text(value).strip(_SHORT_ANSWER_PUNCTUATION).strip()


def parse_confirmation(value: str) -> bool | None:
    """Converte consentimento expresso em linguagem natural sem usar LLM."""
    return parse_flow_answer(value)


def sanitize_user_text(value: str) -> str:
    """Seleciona somente termos nao sensiveis antes de chamar o LLM.

    Usado onde a entrada alimenta classificacao de intencao, nao redacao: ali
    o texto livre nao acrescenta nada e so amplia a superficie de injecao.
    """
    words = re.findall(r"[a-z]+", normalized_text(value))
    return " ".join(word for word in words if word in _SAFE_LLM_WORDS)


def mask_user_text(value: str) -> str:
    """Preserva a pergunta do cliente sem PII, numero ou marcacao estrutural.

    A redacao dos especialistas precisa do texto real para responder no tom de
    quem perguntou. O que nao pode viajar e o dado sensivel: CPF e nascimento
    somem, numeros viram termo neutro e caracteres de controle e de estrutura
    sao descartados para nao abrirem espaco a injecao de instrucao.
    """
    without_control = _CONTROL_PATTERN.sub(" ", value)
    without_pii = _CPF_TEXT_PATTERN.sub(" ", without_control)
    without_pii = _DATE_TEXT_PATTERN.sub(" ", without_pii)
    without_numbers = _NUMBER_TEXT_PATTERN.sub(_MASKED_NUMBER, without_pii)
    collapsed = " ".join(_STRUCTURE_PATTERN.sub(" ", without_numbers).split())
    return collapsed[:_USER_TEXT_LIMIT].strip()


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
    safe_user_text = mask_user_text(user_text) or "pedido bancario"
    messages: list[BaseMessage] = [
        rendered.system_message,
        *_safe_history(recent_messages),
        HumanMessage(
            content=(
                "Redija a resposta final completa a partir do texto validado "
                "abaixo, reconhecendo o que o cliente pediu e respondendo no "
                "tom dele. Preserve cada marcador [DADO_N] exatamente como "
                "está, sem criar fatos, números, decisões ou perguntas novos. "
                f"Pergunta do cliente: {safe_user_text} "
                f"Texto validado: {masked_reply}"
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
    normalized_canonical = normalized_text(canonical_reply)
    if any(leak in normalized_rewritten for leak in _LEAK_PHRASES):
        return False
    if "indisponivel" in normalized_canonical and not any(
        marker in normalized_rewritten
        for marker in ("indisponivel", "nao consegui", "nao foi possivel")
    ):
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
        safe_content = mask_user_text(str(message.content))
        if not safe_content:
            continue
        message_type = AIMessage if isinstance(message, AIMessage) else HumanMessage
        safe_messages.append(message_type(content=safe_content))
    return safe_messages


def format_money(value: Decimal) -> str:
    """Formata valor decimal para exibicao bancaria em pt-BR."""
    formatted = f"{value:,.2f}"
    return formatted.replace(",", "_").replace(".", ",").replace("_", ".")
