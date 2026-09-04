"""Testes dos quatro especialistas e do orcamento deterministico."""

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import TypeVar

import pytest
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from banco_agil.agents._shared import humanize_reply
from banco_agil.agents.credit import handle_credit
from banco_agil.agents.credit_interview import handle_credit_interview
from banco_agil.agents.exchange import handle_exchange
from banco_agil.agents.state import ConversationState, CreditInterviewDraft
from banco_agil.agents.triage import handle_triage
from banco_agil.domain.enums import (
    Agent,
    CreditRequestStatus,
    EmploymentType,
    EndReason,
    Intent,
)
from banco_agil.domain.exceptions import (
    DomainError,
    ExternalServiceUnavailableError,
    RepositoryError,
)
from banco_agil.domain.models import (
    AuthenticationResult,
    Client,
    CpfValidationResult,
    CreditInterview,
    CreditLimitResult,
    ExchangeQuote,
    ExchangeRateResult,
    LimitIncreaseResult,
    ScoreUpdateResult,
)
from banco_agil.services.credit_interview import InterviewField, InterviewProgress

OutputModel = TypeVar("OutputModel", bound=BaseModel)


@dataclass
class RecordingLlm:
    """Registra chamadas e devolve uma resposta estruturada."""

    response: object
    calls: list[tuple[str, list[BaseMessage], str | None]] = field(default_factory=list)

    def calls_remaining(self, turn_id: str) -> int:
        """Informa o saldo de chamadas do turno registrado."""
        used = sum(1 for call in self.calls if call[0] == turn_id)
        return max(0, 2 - used)

    def invoke_structured(
        self,
        turn_id: str,
        messages: list[BaseMessage],
        output_schema: type[OutputModel],
        *,
        prompt_version: str | None = None,
    ) -> OutputModel:
        """Valida a resposta configurada com o schema solicitado."""
        self.calls.append((turn_id, messages, prompt_version))
        return output_schema.model_validate(self.response)


@dataclass
class FakeAuthenticationService:
    """Autentica ou rejeita uma tentativa configurada."""

    client: Client | None
    attempts: int = 0
    calls: list[tuple[str, str]] = field(default_factory=list)
    cpf_calls: list[str] = field(default_factory=list)

    def validate_cpf(
        self,
        state: ConversationState,
        cpf: str,
    ) -> CpfValidationResult:
        """Simula a consulta antecipada do CPF sem autenticar."""
        self.cpf_calls.append(cpf)
        normalized = cpf.replace(".", "").replace("-", "").replace(" ", "")
        if self.client is not None and normalized == self.client.cpf:
            state.pending_cpf = normalized
            return CpfValidationResult(
                valid=True,
                attempts=state.authentication_attempts,
                should_end=False,
            )
        state.authentication_attempts += 1
        return CpfValidationResult(
            valid=False,
            attempts=state.authentication_attempts,
            should_end=state.authentication_attempts >= 3,
        )

    def authenticate(
        self,
        state: ConversationState,
        cpf: str,
        birth_date: str,
    ) -> AuthenticationResult:
        """Atualiza o estado como o servico real faria."""
        self.calls.append((cpf, birth_date))
        if self.client is not None:
            state.authenticated_client = self.client
            state.authentication_attempts = 0
            state.pending_cpf = None
            return AuthenticationResult(
                authenticated=True,
                attempts=0,
                should_end=False,
                client=self.client,
            )
        state.authentication_attempts = self.attempts
        return AuthenticationResult(
            authenticated=False,
            attempts=self.attempts,
            should_end=self.attempts >= 3,
        )


@dataclass
class FakeCreditService:
    """Retorna resultados configurados das operacoes de credito."""

    increase_result: LimitIncreaseResult
    requested_limits: list[Decimal] = field(default_factory=list)
    limit_calls: int = 0

    def get_credit_limit(self, state: ConversationState) -> CreditLimitResult:
        """Registra uma consulta autenticada."""
        del state
        self.limit_calls += 1
        return CreditLimitResult(current_limit=Decimal("2500.00"))

    def request_limit_increase(
        self,
        state: ConversationState,
        new_limit: Decimal,
    ) -> LimitIncreaseResult:
        """Registra o novo limite solicitado."""
        self.requested_limits.append(new_limit)
        if self.increase_result.status is CreditRequestStatus.REJECTED:
            state.requested_limit = new_limit
        return self.increase_result


@dataclass
class FakeInterviewService:
    """Simula progresso incremental da entrevista."""

    progress: InterviewProgress
    starts: list[bool] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)

    def start(
        self,
        state: ConversationState,
        consent: bool,
    ) -> InterviewProgress:
        """Registra o consentimento."""
        self.starts.append(consent)
        if consent:
            state.interview_draft = CreditInterviewDraft(consent_given=True)
        return self.progress

    def collect_answer(
        self,
        state: ConversationState,
        answer: str,
    ) -> InterviewProgress:
        """Registra a resposta recebida."""
        del state
        self.answers.append(answer)
        if self.progress.score_update is not None:
            interview = CreditInterview(
                monthly_income=Decimal("5000.00"),
                employment_type=EmploymentType.FORMAL,
                monthly_expenses=Decimal("2000.00"),
                dependents=1,
                has_active_debts=False,
            )
            return InterviewProgress(
                next_field=None,
                completed_interview=interview,
            )
        return self.progress

    def update_credit_score(
        self,
        state: ConversationState,
        interview: CreditInterview,
    ) -> ScoreUpdateResult:
        """Simula persistencia final por meio da tool."""
        del interview
        state.active_agent = Agent.CREDIT
        state.credit_reanalysis_pending = True
        if self.progress.score_update is None:
            raise AssertionError("score update was not configured")
        return self.progress.score_update


@dataclass
class FakeExchangeService:
    """Retorna cotacao ou falha externa configurada."""

    result: ExchangeRateResult | None
    calls: list[tuple[str, str]] = field(default_factory=list)

    def get_exchange_rate(
        self,
        state: ConversationState,
        base_currency: str,
        quote_currency: str,
    ) -> ExchangeRateResult:
        """Registra o par solicitado."""
        del state
        self.calls.append((base_currency, quote_currency))
        if self.result is None:
            raise ExternalServiceUnavailableError("provider unavailable")
        return self.result


@pytest.fixture
def client() -> Client:
    """Cria cliente ficticio autenticado."""
    return Client(
        cpf="01234567890",
        birth_date=date(1990, 5, 20),
        credit_limit=Decimal("2500.00"),
        credit_score=700,
    )


def _increase_result(status: CreditRequestStatus) -> LimitIncreaseResult:
    return LimitIncreaseResult(
        current_limit=Decimal("2500.00"),
        requested_limit=Decimal("4000.00"),
        status=status,
        offer_interview=status is CreditRequestStatus.REJECTED,
    )


@pytest.mark.parametrize(
    "agent", [Agent.CREDIT, Agent.CREDIT_INTERVIEW, Agent.EXCHANGE]
)
def test_humanization_rewrites_reply_without_exposing_data(
    client: Client,
    agent: Agent,
) -> None:
    state = ConversationState(authenticated_client=client, active_agent=agent)
    llm = RecordingLlm({"reply": "Claro! Resposta canônica com [DADO_1]."})

    reply = humanize_reply(
        state,
        "Resposta canônica com R$ 2.500,00.",
        llm,
        "humanize-turn",
        responding_agent=agent,
        recent_messages=[],
        user_text="Meu CPF é 01234567890 e quero consultar meu limite",
    )

    assert reply == "Claro! Resposta canônica com R$ 2.500,00."
    _, messages, version = llm.calls[0]
    expected_version = "1.4.0" if agent is Agent.EXCHANGE else "1.3.0"
    assert version == f"global@1.3.0+{agent.value}@{expected_version}"
    assert "01234567890" not in str(messages)
    assert "2.500,00" not in str(messages)
    assert "R$ 2.500,00" not in str(messages)


def test_humanization_violation_preserves_canonical_reply(
    client: Client,
) -> None:
    state = ConversationState(authenticated_client=client)
    llm = RecordingLlm({"reply": "Aprovado com R$ 9.999,00 garantidos."})
    canonical = "Resposta canônica com R$ 2.500,00."

    reply = humanize_reply(
        state,
        canonical,
        llm,
        "humanize-turn",
        responding_agent=Agent.CREDIT,
        recent_messages=[],
        user_text="consultar limite",
    )

    assert reply == canonical


def test_triage_collects_credentials_one_at_a_time(client: Client) -> None:
    state = ConversationState()
    service = FakeAuthenticationService(client)

    cpf_reply = handle_triage(state, "012.345.678-90", service)
    auth_reply = handle_triage(state, "20/05/1990", service)

    assert "nascimento" in cpf_reply.casefold()
    assert "DD/MM/AAAA" in cpf_reply
    assert "ajudar" in auth_reply.casefold()
    assert service.calls == [("01234567890", "20/05/1990")]
    assert state.authenticated_client is client


def test_triage_explains_validation_before_requesting_cpf(client: Client) -> None:
    state = ConversationState()

    reply = handle_triage(
        state,
        "Quero saber se posso aumentar meu limite",
        FakeAuthenticationService(client),
    )

    assert "antes" in reply.casefold()
    assert "validar" in reply.casefold()
    assert "cpf" in reply.casefold()
    assert state.authenticated_client is None
    assert state.authentication_attempts == 0


def test_initial_request_with_value_is_not_counted_as_invalid_cpf(
    client: Client,
) -> None:
    state = ConversationState()

    reply = handle_triage(
        state,
        "Quero aumentar meu limite para 5000 reais",
        FakeAuthenticationService(client),
    )

    assert "validar alguns dados" in reply.casefold()
    assert state.authentication_attempts == 0


def test_triage_third_failure_ends_without_disclosing_wrong_field() -> None:
    state = ConversationState(pending_cpf="01234567890")
    reply = handle_triage(state, "20/05/1990", FakeAuthenticationService(None, 3))

    assert state.ended
    assert state.end_reason is EndReason.AUTHENTICATION_FAILURES
    assert "cpf" not in reply.casefold()
    assert "nascimento" not in reply.casefold()


@pytest.mark.parametrize(
    "user_text",
    (
        "quero aumentar meu limite",
        "quero alterar o meu limite",
        "preciso mudar meu limite",
        "gostaria de ajustar o limite",
        "quero modificar meu limite",
    ),
)
def test_triage_routes_increase_synonyms_without_llm(
    client: Client,
    user_text: str,
) -> None:
    state = ConversationState(authenticated_client=client)
    llm = RecordingLlm({"intent": "other"})

    reply = handle_triage(
        state,
        user_text,
        FakeAuthenticationService(client),
        llm=llm,
        turn_id="clear-turn",
    )

    assert reply
    assert state.intent is Intent.LIMIT_INCREASE
    assert state.active_agent is Agent.CREDIT
    assert llm.calls == []


@pytest.mark.parametrize(
    "user_text",
    (
        "quero aumentar meu score",
        "como está a análise do meu score?",
        "quero fazer a entrevista de crédito",
    ),
)
def test_triage_routes_score_review_to_interview_without_llm(
    client: Client,
    user_text: str,
) -> None:
    state = ConversationState(authenticated_client=client)
    llm = RecordingLlm({"intent": "other"})

    handle_triage(
        state,
        user_text,
        FakeAuthenticationService(client),
        llm=llm,
        turn_id="score-turn",
    )

    assert state.intent is Intent.CREDIT_INTERVIEW
    assert state.active_agent is Agent.CREDIT_INTERVIEW
    assert llm.calls == []


def test_triage_uses_llm_only_for_ambiguous_authenticated_intent(
    client: Client,
) -> None:
    state = ConversationState(authenticated_client=client)
    llm = RecordingLlm({"intent": "exchange_rate"})

    handle_triage(
        state,
        "preciso resolver uma coisa do exterior",
        FakeAuthenticationService(client),
        llm=llm,
        turn_id="ambiguous-turn",
    )

    assert state.intent is Intent.EXCHANGE_RATE
    assert state.active_agent is Agent.EXCHANGE
    assert len(llm.calls) == 1
    _, messages, version = llm.calls[0]
    assert version == "global@1.3.0+triage@1.5.0"
    assert "exterior" in str(messages[-1].content)


def test_triage_keeps_ambiguous_intent_in_triage(client: Client) -> None:
    state = ConversationState(authenticated_client=client)

    reply = handle_triage(state, "preciso de ajuda", FakeAuthenticationService(client))

    assert "limite" in reply.casefold()
    assert "cotação" in reply.casefold()
    assert state.active_agent is Agent.TRIAGE


def test_humanization_does_not_call_llm_for_triage(client: Client) -> None:
    state = ConversationState(authenticated_client=client)
    llm = RecordingLlm({"reply": "Texto alterado."})
    canonical = "Posso ajudar com limite ou cotação?"

    reply = humanize_reply(
        state,
        canonical,
        llm,
        "turn-2",
        responding_agent=Agent.TRIAGE,
        recent_messages=[],
        user_text="preciso resolver outra coisa",
    )

    assert reply == canonical
    assert llm.calls == []


def test_authentication_repository_failure_returns_controlled_reply() -> None:
    class FailingAuthenticationService(FakeAuthenticationService):
        def authenticate(
            self,
            state: ConversationState,
            cpf: str,
            birth_date: str,
        ) -> AuthenticationResult:
            del state, cpf, birth_date
            raise RepositoryError("sensitive technical detail")

    state = ConversationState(pending_cpf="01234567890")

    reply = handle_triage(
        state,
        "20/05/1990",
        FailingAuthenticationService(None),
    )

    assert "tente novamente" in reply.casefold()
    assert "technical" not in reply


def test_credit_consults_limit_without_llm(client: Client) -> None:
    state = ConversationState(
        authenticated_client=client,
        active_agent=Agent.CREDIT,
        intent=Intent.CREDIT_LIMIT,
    )
    service = FakeCreditService(_increase_result(CreditRequestStatus.APPROVED))

    reply = handle_credit(state, "qual é meu limite?", service)

    assert "2.500,00" in reply
    assert service.limit_calls == 1
    assert service.requested_limits == []


def test_rejected_increase_offers_interview_without_promise(client: Client) -> None:
    state = ConversationState(
        authenticated_client=client,
        active_agent=Agent.CREDIT,
        intent=Intent.LIMIT_INCREASE,
    )
    service = FakeCreditService(_increase_result(CreditRequestStatus.REJECTED))

    reply = handle_credit(state, "R$ 4.000,00", service)

    assert service.requested_limits == [Decimal("4000.00")]
    assert state.active_agent is Agent.CREDIT_INTERVIEW
    assert "4.000,00" in reply
    assert "não pôde ser aprovado" in reply.casefold()
    assert "entrevista" in reply.casefold()
    assert "aprovação garantida" not in reply.casefold()


def test_credit_reanalysis_reuses_requested_limit(client: Client) -> None:
    state = ConversationState(
        authenticated_client=client,
        active_agent=Agent.CREDIT,
        intent=Intent.LIMIT_INCREASE,
        requested_limit=Decimal("4000.00"),
        credit_reanalysis_pending=True,
    )
    service = FakeCreditService(_increase_result(CreditRequestStatus.APPROVED))

    reply = handle_credit(state, "", service)

    assert service.requested_limits == [Decimal("4000.00")]
    assert not state.credit_reanalysis_pending
    assert "4.000,00" in reply
    assert "aprovado" in reply.casefold()


def test_credit_asks_for_desired_total_with_context(client: Client) -> None:
    state = ConversationState(
        authenticated_client=client,
        active_agent=Agent.CREDIT,
        intent=Intent.LIMIT_INCREASE,
    )

    reply = handle_credit(
        state,
        "quero aumentar meu limite",
        FakeCreditService(_increase_result(CreditRequestStatus.APPROVED)),
    )

    assert "posso analisar" in reply.casefold()
    assert "limite total" in reply.casefold()


def test_credit_repository_failure_returns_controlled_reply(client: Client) -> None:
    class FailingCreditService(FakeCreditService):
        def request_limit_increase(
            self,
            state: ConversationState,
            new_limit: Decimal,
        ) -> LimitIncreaseResult:
            del state, new_limit
            raise RepositoryError("sensitive technical detail")

    state = ConversationState(
        authenticated_client=client,
        active_agent=Agent.CREDIT,
        intent=Intent.LIMIT_INCREASE,
    )

    reply = handle_credit(
        state,
        "4000",
        FailingCreditService(_increase_result(CreditRequestStatus.APPROVED)),
    )

    assert "tente novamente" in reply.casefold()
    assert "technical" not in reply


def test_interview_requires_consent_before_collecting(client: Client) -> None:
    state = ConversationState(
        authenticated_client=client,
        active_agent=Agent.CREDIT_INTERVIEW,
        requested_limit=Decimal("4000.00"),
    )
    service = FakeInterviewService(
        InterviewProgress(next_field=InterviewField.MONTHLY_INCOME)
    )

    unclear_reply = handle_credit_interview(state, "talvez", service)
    accepted_reply = handle_credit_interview(state, "sim", service)

    assert "sim ou não" in unclear_reply.casefold()
    assert "renda mensal" in accepted_reply.casefold()
    assert service.starts == [True]
    assert service.answers == []


def test_interview_completion_returns_to_credit(client: Client) -> None:
    score = ScoreUpdateResult(previous_score=700, new_score=760)
    state = ConversationState(
        authenticated_client=client,
        active_agent=Agent.CREDIT_INTERVIEW,
        requested_limit=Decimal("4000.00"),
        interview_draft=CreditInterviewDraft(consent_given=True),
    )
    service = FakeInterviewService(
        InterviewProgress(next_field=None, score_update=score)
    )

    reply = handle_credit_interview(state, "não", service)

    assert service.answers == ["não"]
    assert state.active_agent is Agent.CREDIT
    assert state.credit_reanalysis_pending
    assert "reanálise" in reply.casefold()
    assert "aprovação garantida" not in reply.casefold()


def test_invalid_interview_answer_repeats_current_question(client: Client) -> None:
    state = ConversationState(
        authenticated_client=client,
        active_agent=Agent.CREDIT_INTERVIEW,
        requested_limit=Decimal("4000.00"),
        interview_draft=CreditInterviewDraft(consent_given=True),
    )

    class InvalidInterviewService(FakeInterviewService):
        def collect_answer(
            self,
            state: ConversationState,
            answer: str,
        ) -> InterviewProgress:
            del state, answer
            raise DomainError("invalid interview answer")

    service = InvalidInterviewService(
        InterviewProgress(next_field=InterviewField.MONTHLY_INCOME)
    )

    reply = handle_credit_interview(state, "valor inválido", service)

    assert "renda mensal" in reply.casefold()
    assert "formato" in reply.casefold()


def test_score_repository_failure_returns_controlled_reply(client: Client) -> None:
    score = ScoreUpdateResult(previous_score=700, new_score=760)

    class FailingInterviewService(FakeInterviewService):
        def update_credit_score(
            self,
            state: ConversationState,
            interview: CreditInterview,
        ) -> ScoreUpdateResult:
            del state, interview
            raise RepositoryError("sensitive technical detail")

    state = ConversationState(
        authenticated_client=client,
        active_agent=Agent.CREDIT_INTERVIEW,
        requested_limit=Decimal("4000.00"),
        interview_draft=CreditInterviewDraft(consent_given=True),
    )

    reply = handle_credit_interview(
        state,
        "não",
        FailingInterviewService(InterviewProgress(next_field=None, score_update=score)),
    )

    assert "tente novamente" in reply.casefold()
    assert "technical" not in reply


def test_exchange_maps_dollar_and_returns_confirmed_quote(client: Client) -> None:
    quote = ExchangeQuote(
        base_currency="USD",
        quote_currency="BRL",
        rate=Decimal("5.25"),
        source="AwesomeAPI",
        quoted_at=datetime(2026, 9, 3, 12, 0, tzinfo=UTC),
    )
    service = FakeExchangeService(ExchangeRateResult(quote=quote))
    state = ConversationState(
        authenticated_client=client,
        active_agent=Agent.EXCHANGE,
        intent=Intent.EXCHANGE_RATE,
    )

    reply = handle_exchange(state, "cotação do dólar", service)

    assert service.calls == [("USD", "BRL")]
    assert all(
        value in reply
        for value in ("🇺🇸", "dólar", "5,25", "09:00", "Brasília", "AwesomeAPI")
    )
    assert "USD-BRL" not in reply


@pytest.mark.parametrize(
    ("user_text", "expected_pair", "expected_markers"),
    (
        ("BRL-USD", ("BRL", "USD"), ("🇧🇷", "BRL-USD")),
        ("EUR-BRL", ("EUR", "BRL"), ("🇪🇺", "euro")),
        ("qual o valor do euro hoje?", ("EUR", "BRL"), ("🇪🇺", "euro")),
        ("qual o valor do dólar hoje?", ("USD", "BRL"), ("🇺🇸", "dólar")),
    ),
)
def test_exchange_accepts_free_pair_forms(
    client: Client,
    user_text: str,
    expected_pair: tuple[str, str],
    expected_markers: tuple[str, str],
) -> None:
    quote = ExchangeQuote(
        base_currency=expected_pair[0],
        quote_currency=expected_pair[1],
        rate=Decimal("5.25"),
        source="AwesomeAPI",
        quoted_at=datetime(2026, 9, 3, 12, 0, tzinfo=UTC),
    )
    service = FakeExchangeService(ExchangeRateResult(quote=quote))
    state = ConversationState(
        authenticated_client=client,
        active_agent=Agent.EXCHANGE,
        intent=Intent.EXCHANGE_RATE,
    )

    reply = handle_exchange(state, user_text, service)

    assert service.calls == [expected_pair]
    assert all(marker in reply for marker in expected_markers)
    assert "09:00" in reply
    assert "Brasília" in reply
    assert "AwesomeAPI" in reply


def test_exchange_failure_never_invents_rate(client: Client) -> None:
    state = ConversationState(
        authenticated_client=client,
        active_agent=Agent.EXCHANGE,
    )

    reply = handle_exchange(state, "USD-BRL", FakeExchangeService(None))

    assert "indisponível" in reply.casefold()
    assert "tente novamente" in reply.casefold()
    assert not any(character.isdigit() for character in reply)


def test_exchange_understands_named_cross_currency_pair(client: Client) -> None:
    quote = ExchangeQuote(
        base_currency="EUR",
        quote_currency="USD",
        rate=Decimal("1.10"),
        source="AwesomeAPI",
        quoted_at=datetime(2026, 9, 3, 12, 0, tzinfo=UTC),
    )
    service = FakeExchangeService(ExchangeRateResult(quote=quote))
    state = ConversationState(
        authenticated_client=client,
        active_agent=Agent.EXCHANGE,
    )

    handle_exchange(state, "euro para dólar", service)

    assert service.calls == [("EUR", "USD")]


def test_exchange_rejects_same_currency_without_calling_provider(
    client: Client,
) -> None:
    service = FakeExchangeService(None)
    state = ConversationState(
        authenticated_client=client,
        active_agent=Agent.EXCHANGE,
    )

    reply = handle_exchange(state, "USD-USD", service)

    assert "diferentes" in reply.casefold()
    assert service.calls == []


@pytest.mark.parametrize("handler_name", ["credit", "interview", "exchange"])
def test_specialists_redirect_unauthenticated_sessions(handler_name: str) -> None:
    state = ConversationState(
        active_agent=Agent(
            handler_name if handler_name != "interview" else "credit_interview"
        )
    )
    if handler_name == "credit":
        reply = handle_credit(
            state,
            "limite",
            FakeCreditService(_increase_result(CreditRequestStatus.APPROVED)),
        )
    elif handler_name == "interview":
        reply = handle_credit_interview(
            state,
            "sim",
            FakeInterviewService(InterviewProgress(next_field=None)),
        )
    else:
        reply = handle_exchange(state, "USD-BRL", FakeExchangeService(None))

    assert state.active_agent is Agent.TRIAGE
    assert "confirmar seus dados" in reply.casefold()


def test_end_request_has_priority_in_every_specialist(client: Client) -> None:
    states = [
        ConversationState(authenticated_client=client, active_agent=agent)
        for agent in Agent
    ]

    replies = [
        handle_triage(
            states[0],
            "por favor, encerre o atendimento",
            FakeAuthenticationService(client),
        ),
        handle_credit(
            states[1],
            "encerrar",
            FakeCreditService(_increase_result(CreditRequestStatus.APPROVED)),
        ),
        handle_credit_interview(
            states[2],
            "encerrar",
            FakeInterviewService(InterviewProgress(next_field=None)),
        ),
        handle_exchange(states[3], "encerrar", FakeExchangeService(None)),
    ]

    assert all(state.ended for state in states)
    assert all(state.end_reason is EndReason.USER_REQUEST for state in states)
    assert all("encerrado" in reply.casefold() for reply in replies)


def test_end_parser_does_not_stop_an_unfinished_credit_request(client: Client) -> None:
    state = ConversationState(
        authenticated_client=client,
        active_agent=Agent.CREDIT,
        intent=Intent.LIMIT_INCREASE,
    )

    reply = handle_credit(
        state,
        "quero finalizar minha solicitação de aumento",
        FakeCreditService(_increase_result(CreditRequestStatus.APPROVED)),
    )

    assert not state.ended
    assert "limite total" in reply.casefold()
