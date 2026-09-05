"""Entendimento do turno pelo LLM, aterrado no texto antes de ser usado.

O Groq entende; o Python decide. Cada campo extraido so vale se o texto do
cliente o sustenta: valor e dependentes precisam aparecer como digito ou
palavra, moeda precisa ser suportada, topico precisa existir no catalogo e a
pergunta de esclarecimento precisa passar pelas mesmas guardas da redacao.
Sem credencial ou em falha, cada especialista volta ao seu parser e a sua
pergunta canonica.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, ConfigDict

from banco_agil.agents._shared import (
    RESUMABLE_INTENTS,
    FlowChange,
    mask_user_text,
    normalize_dashes,
    normalized_text,
    parse_flow_change,
    safe_history,
)
from banco_agil.domain.enums import EmploymentType, Intent
from banco_agil.domain.exceptions import IntegrationError
from banco_agil.integrations.llm import MAX_CALLS_PER_TURN, StructuredLlm
from banco_agil.knowledge.catalog import KNOWLEDGE_CATALOG
from banco_agil.prompts.renderer import render_understanding_prompt

SUPPORTED_CURRENCIES = frozenset(
    {"USD", "EUR", "GBP", "BRL", "ARS", "JPY", "CHF", "CAD", "AUD", "CNY", "BTC"}
)
_KNOWLEDGE_TOPICS = tuple(entry.key for entry in KNOWLEDGE_CATALOG)
_FLOW_LABELS = {
    Intent.CREDIT_LIMIT: "consulta de limite de crédito",
    Intent.LIMIT_INCREASE: "pedido de aumento de limite, aguardando o novo limite",
    Intent.CREDIT_INTERVIEW: "entrevista de crédito, com perguntas sobre renda",
    Intent.EXCHANGE_RATE: "consulta de cotação, aguardando a moeda",
    Intent.INFORMATION: "dúvida sobre como o atendimento funciona",
    Intent.UNKNOWN: "escolha do serviço, logo após a autenticação",
}
_NUMBER_PATTERN = re.compile(r"\d[\d.,]*")
_THOUSAND_PATTERN = re.compile(r"\b(\d[\d.,]*)\s*(?:mil|k)\b")
_NUMBER_WORDS = {
    "zero": 0,
    "nenhum": 0,
    "nenhuma": 0,
    "um": 1,
    "uma": 1,
    "dois": 2,
    "duas": 2,
    "tres": 3,
    "quatro": 4,
    "cinco": 5,
    "seis": 6,
    "sete": 7,
    "oito": 8,
    "nove": 9,
    "dez": 10,
}
_CLARIFICATION_LIMIT = 240
_CLARIFICATION_MAX_SENTENCES = 3
_LEAK_PHRASES = (
    "system prompt",
    "prompt do sistema",
    "instrucao",
    "como modelo de linguagem",
    "regras internas",
    "texto validado",
    "passo atual",
)


class LlmUnderstanding(BaseModel):
    """Schema devolvido pelo modelo; nada daqui e usado sem aterramento."""

    model_config = ConfigDict(extra="ignore")

    intent: Intent = Intent.UNKNOWN
    declines_current: bool = False
    amount: str | None = None
    base_currency: str | None = None
    quote_currency: str | None = None
    monthly_income: str | None = None
    employment_type: EmploymentType | None = None
    monthly_expenses: str | None = None
    dependents: int | None = None
    has_active_debts: bool | None = None
    knowledge_topic: str | None = None
    clarification: str | None = None


class TurnUnderstanding(BaseModel):
    """Leitura do turno ja validada contra o texto do cliente."""

    model_config = ConfigDict(frozen=True)

    intent: Intent = Intent.UNKNOWN
    declines_current: bool = False
    amount: Decimal | None = None
    currency_pair: tuple[str, str] | None = None
    monthly_income: Decimal | None = None
    employment_type: EmploymentType | None = None
    monthly_expenses: Decimal | None = None
    dependents: int | None = None
    has_active_debts: bool | None = None
    knowledge_topic: str | None = None
    clarification: str | None = None

    def flow_change(self, current_intent: Intent) -> FlowChange | None:
        """Traduz intencao e recusa em mudanca de fluxo, ou None.

        Encerrar continua exigindo pedido explicito e deterministico: aqui
        `end_service` vale como recusa. Repetir o proprio fluxo nao e mudanca.
        """
        if self.intent in RESUMABLE_INTENTS and self.intent is not current_intent:
            return FlowChange(requested_intent=self.intent)
        if self.intent is Intent.HELP:
            return FlowChange(requested_intent=Intent.HELP)
        if self.declines_current or self.intent is Intent.END_SERVICE:
            return FlowChange(declines_current=True)
        return None


def ground_understanding(raw: LlmUnderstanding, user_text: str) -> TurnUnderstanding:
    """Mantem apenas o que o texto do cliente sustenta."""
    pair = None
    if (
        raw.base_currency in SUPPORTED_CURRENCIES
        and raw.quote_currency in SUPPORTED_CURRENCIES
        and raw.base_currency != raw.quote_currency
    ):
        pair = (raw.base_currency, raw.quote_currency)
    return TurnUnderstanding(
        intent=raw.intent,
        declines_current=raw.declines_current,
        amount=_grounded_money(raw.amount, user_text),
        currency_pair=pair,
        monthly_income=_grounded_money(raw.monthly_income, user_text),
        employment_type=raw.employment_type,
        monthly_expenses=_grounded_money(raw.monthly_expenses, user_text),
        dependents=_grounded_small_int(raw.dependents, user_text),
        has_active_debts=raw.has_active_debts,
        knowledge_topic=(
            raw.knowledge_topic if raw.knowledge_topic in _KNOWLEDGE_TOPICS else None
        ),
        clarification=accept_clarification(raw.clarification),
    )


def numbers_in_text(user_text: str) -> set[Decimal]:
    """Le todos os valores que o cliente escreveu, com "mil" e "k" expandidos."""
    normalized = normalized_text(user_text)
    found: set[Decimal] = set()
    for match in _THOUSAND_PATTERN.finditer(normalized):
        value = _parse_decimal(match.group(1))
        if value is not None:
            found.add(value * 1000)
    for match in _NUMBER_PATTERN.finditer(normalized):
        value = _parse_decimal(match.group())
        if value is not None:
            found.add(value)
    return found


def _parse_decimal(token: str) -> Decimal | None:
    value = token.strip(".,")
    if "," in value:
        value = value.replace(".", "").replace(",", ".")
    elif value.count(".") == 1 and len(value.rsplit(".", maxsplit=1)[1]) == 3:
        value = value.replace(".", "")
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        return None
    return parsed if parsed.is_finite() else None


def _grounded_money(value: str | None, user_text: str) -> Decimal | None:
    if value is None:
        return None
    parsed = _parse_decimal(value.strip())
    if parsed is None or parsed < 0:
        return None
    return parsed if parsed in numbers_in_text(user_text) else None


def _grounded_small_int(value: int | None, user_text: str) -> int | None:
    if value is None or value < 0:
        return None
    if Decimal(value) in numbers_in_text(user_text):
        return value
    words = set(normalized_text(user_text).split())
    if any(_NUMBER_WORDS.get(word) == value for word in words):
        return value
    return None


def accept_clarification(text: str | None) -> str | None:
    """Aceita a pergunta de esclarecimento so dentro das guardas da redacao.

    Sem digito, para nao inventar valor; curta e terminando em pergunta, para
    continuar sendo pergunta; sem vazamento de instrucao.
    """
    if not text:
        return None
    cleaned = " ".join(normalize_dashes(text).split())
    if not cleaned.endswith("?") or len(cleaned) > _CLARIFICATION_LIMIT:
        return None
    if re.search(r"\d", cleaned):
        return None
    if len(re.findall(r"[.!?]", cleaned)) > _CLARIFICATION_MAX_SENTENCES:
        return None
    normalized = normalized_text(cleaned)
    if any(phrase in normalized for phrase in _LEAK_PHRASES):
        return None
    return cleaned


def resolve_context(
    context: "TurnContext | None",
    llm: StructuredLlm | None,
    turn_id: str,
    recent_messages: Sequence[BaseMessage],
) -> "TurnContext":
    """Aceita o contexto do grafo ou monta um a partir dos argumentos soltos."""
    if context is not None:
        return context
    return TurnContext(llm=llm, turn_id=turn_id, recent_messages=recent_messages)


@dataclass
class TurnContext:
    """Recursos de um turno compartilhados pelos nos que ele atravessa.

    O entendimento e memorizado: quem recebe o turno de outro no reaproveita
    a leitura em vez de gastar outra chamada. `text_classified` marca que a
    triagem ja roteou este texto por parser, e entao nenhum especialista pode
    reinterpreta-lo como troca de fluxo.
    """

    llm: StructuredLlm | None = None
    turn_id: str = ""
    recent_messages: Sequence[BaseMessage] = ()
    text_classified: bool = False
    _understanding: TurnUnderstanding | None = field(default=None, init=False)
    _attempted: bool = field(default=False, init=False)

    def understand(self, user_text: str, flow: Intent) -> TurnUnderstanding | None:
        """Pede ao LLM a leitura do turno, uma unica vez por turno.

        Usa uma chamada do orcamento e deixa a outra para a redacao; sem saldo,
        sem credencial ou em falha devolve None e o especialista segue com o
        parser.
        """
        if self._attempted:
            return self._understanding
        if self.text_classified:
            # A triagem ja roteou este texto por parser; reler nao acrescenta
            # nada e custaria a chamada que a redacao precisa.
            return None
        self._attempted = True
        llm = self.llm
        if (
            llm is None
            or not self.turn_id
            or llm.calls_remaining(self.turn_id) < MAX_CALLS_PER_TURN
        ):
            return None
        safe_user_text = mask_user_text(user_text)
        if not safe_user_text:
            return None
        rendered = render_understanding_prompt(
            _FLOW_LABELS.get(flow, "atendimento"), _KNOWLEDGE_TOPICS
        )
        messages: list[BaseMessage] = [
            rendered.system_message,
            *safe_history(self.recent_messages),
            HumanMessage(content=safe_user_text),
        ]
        try:
            raw = llm.invoke_structured(
                self.turn_id,
                messages,
                LlmUnderstanding,
                prompt_version=rendered.prompt_version,
            )
        except IntegrationError:
            return None
        self._understanding = ground_understanding(raw, user_text)
        return self._understanding

    def flow_change(self, user_text: str, current_intent: Intent) -> FlowChange | None:
        """Parser primeiro, LLM depois."""
        change = parse_flow_change(user_text, current_intent)
        if change is not None:
            return change
        understanding = self.understand(user_text, current_intent)
        if understanding is None:
            return None
        return understanding.flow_change(current_intent)

    def clarification(self, user_text: str, flow: Intent) -> str | None:
        """Pergunta de esclarecimento redigida pelo LLM, se houver e for segura."""
        understanding = self.understand(user_text, flow)
        return understanding.clarification if understanding is not None else None
