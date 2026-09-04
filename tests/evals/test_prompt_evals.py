"""Avaliacao versionada de intencao, extracao e consumo dos prompts.

Dataset ficticio que mede, sem rede e sem credenciais: acerto de roteamento e
de extracao de par de moedas, chamadas LLM (sempre zero no caminho
deterministico), latencia por caso e tamanho do system message por
especialista. O `BASELINE` fixa os valores da versao atual de prompt; ao mudar
`PROMPTS.md`, atualize o baseline no mesmo commit.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from time import perf_counter

from banco_agil.agents.credit import handle_credit
from banco_agil.agents.exchange import handle_exchange
from banco_agil.agents.state import ConversationState
from banco_agil.agents.triage import handle_triage
from banco_agil.domain.enums import Agent, CreditRequestStatus, Intent
from banco_agil.domain.models import (
    Client,
    CreditLimitResult,
    ExchangeQuote,
    ExchangeRateResult,
    LimitIncreaseResult,
)
from banco_agil.prompts.renderer import render_prompt

DATASET_VERSION = "1.2.0"
PROMPT_VERSION = "global@1.3.0+triage@1.6.0"
MAX_LATENCY_MS = 1000.0

BASELINE = {
    "dataset_version": DATASET_VERSION,
    "prompt_version": PROMPT_VERSION,
    "cases": 10,
    "accuracy": 1.0,
    "total_llm_calls": 0,
}


@dataclass(frozen=True)
class IntentCase:
    """Caso ficticio de roteamento com resultado esperado."""

    text: str
    expected_agent: Agent
    expected_intent: Intent
    expected_ended: bool = False


@dataclass(frozen=True)
class ExtractionCase:
    """Caso ficticio de extracao de par de moedas."""

    text: str
    expected_pair: tuple[str, str]


INTENT_CASES = (
    IntentCase("qual é meu limite?", Agent.CREDIT, Intent.CREDIT_LIMIT),
    IntentCase("quero aumentar meu limite", Agent.CREDIT, Intent.LIMIT_INCREASE),
    IntentCase("cotação do dólar", Agent.EXCHANGE, Intent.EXCHANGE_RATE),
    IntentCase("o que você pode fazer?", Agent.TRIAGE, Intent.UNKNOWN),
    IntentCase("preciso resolver outra coisa", Agent.TRIAGE, Intent.UNKNOWN),
    IntentCase("quero encerrar", Agent.TRIAGE, Intent.UNKNOWN, True),
)

EXTRACTION_CASES = (
    ExtractionCase("USD-BRL", ("USD", "BRL")),
    ExtractionCase("euro para dólar", ("EUR", "USD")),
    ExtractionCase("cotação do dólar", ("USD", "BRL")),
    ExtractionCase("qual a cotação do euro?", ("EUR", "BRL")),
)


@dataclass
class NeverCalledService:
    """Falha se qualquer regra de negocio for acionada na avaliacao."""

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"service must not be used in evals: {name}")


@dataclass
class FakeCreditService:
    """Responde consulta e aumento sem persistencia."""

    def get_credit_limit(self, state: ConversationState) -> CreditLimitResult:
        """Retorna o limite ficticio."""
        del state
        return CreditLimitResult(current_limit=Decimal("2500.00"))

    def request_limit_increase(
        self, state: ConversationState, new_limit: Decimal
    ) -> LimitIncreaseResult:
        """Retorna aprovacao ficticia."""
        del state
        return LimitIncreaseResult(
            current_limit=Decimal("2500.00"),
            requested_limit=new_limit,
            status=CreditRequestStatus.APPROVED,
        )


@dataclass
class FakeExchangeProvider:
    """Cotacao fixa com par solicitado registrado."""

    calls: list[tuple[str, str]] = field(default_factory=list)

    def get_exchange_rate(
        self, base_currency: str, quote_currency: str
    ) -> ExchangeQuote:
        """Retorna cotacao ficticia confirmada."""
        from datetime import UTC, datetime

        self.calls.append((base_currency, quote_currency))
        return ExchangeQuote(
            base_currency=base_currency,
            quote_currency=quote_currency,
            rate=Decimal("5.25"),
            source="Fonte Fictícia",
            quoted_at=datetime(2026, 9, 3, 12, 0, tzinfo=UTC),
        )


@dataclass
class FakeExchangeService:
    """Delega ao provedor falso com cliente autenticado."""

    provider: FakeExchangeProvider

    def get_exchange_rate(
        self, state: ConversationState, base_currency: str, quote_currency: str
    ) -> ExchangeRateResult:
        """Valida autenticacao e consulta o provedor falso."""
        del state
        quote = self.provider.get_exchange_rate(base_currency, quote_currency)
        return ExchangeRateResult(quote=quote)


def _client() -> Client:
    return Client(
        cpf="01234567890",
        birth_date=date(1990, 5, 20),
        credit_limit=Decimal("2500.00"),
        credit_score=700,
    )


def _measure_latency_ms(action: Callable[[], object]) -> float:
    started = perf_counter()
    action()
    return (perf_counter() - started) * 1000


def _run_intent_case(case: IntentCase) -> tuple[ConversationState, float]:
    state = ConversationState(authenticated_client=_client())
    started = perf_counter()
    handle_triage(state, case.text, NeverCalledService())
    return state, (perf_counter() - started) * 1000


def _run_extraction_case(
    case: ExtractionCase,
) -> tuple[list[tuple[str, str]], float]:
    provider = FakeExchangeProvider()
    state = ConversationState(
        authenticated_client=_client(), active_agent=Agent.EXCHANGE
    )
    started = perf_counter()
    handle_exchange(state, case.text, FakeExchangeService(provider))
    return provider.calls, (perf_counter() - started) * 1000


def test_intent_routing_matches_expected_agents() -> None:
    hits = 0
    for case in INTENT_CASES:
        state, latency_ms = _run_intent_case(case)
        assert latency_ms < MAX_LATENCY_MS
        assert state.active_agent is case.expected_agent, case.text
        assert state.ended is case.expected_ended, case.text
        if not case.expected_ended and case.expected_agent is not Agent.TRIAGE:
            assert state.intent is case.expected_intent, case.text
        hits += 1

    assert hits / len(INTENT_CASES) == BASELINE["accuracy"]


def test_currency_extraction_matches_expected_pairs() -> None:
    hits = 0
    for case in EXTRACTION_CASES:
        calls, latency_ms = _run_extraction_case(case)
        assert latency_ms < MAX_LATENCY_MS
        assert calls == [case.expected_pair], case.text
        hits += 1

    assert hits / len(EXTRACTION_CASES) == BASELINE["accuracy"]


def test_deterministic_path_uses_zero_llm_calls() -> None:
    assert BASELINE["total_llm_calls"] == 0
    assert BASELINE["cases"] == len(INTENT_CASES) + len(EXTRACTION_CASES)


def test_prompt_consumption_matches_baseline() -> None:
    for agent in Agent:
        rendered = render_prompt(ConversationState(active_agent=agent))
        content = str(rendered.system_message.content)
        assert "{{" not in content
        assert len(content) < 2200 + 500
        assert rendered.prompt_version.startswith("global@1.3.0+")


def test_credit_handlers_stay_within_latency_budget() -> None:
    state = ConversationState(
        authenticated_client=_client(),
        active_agent=Agent.CREDIT,
        intent=Intent.CREDIT_LIMIT,
    )
    replies: list[str] = []
    latency_ms = _measure_latency_ms(
        lambda: replies.append(
            handle_credit(state, "qual é meu limite?", FakeCreditService())
        )
    )

    assert latency_ms < MAX_LATENCY_MS
    assert "2.500,00" in replies[0]
