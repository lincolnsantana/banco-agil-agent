"""Testes do estado e das acoes da interface Streamlit."""

import sys
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage  # noqa: E402

import app  # noqa: E402
from app import (  # noqa: E402
    CHAT_VIEW,
    LANDING_VIEW,
    QuickAction,
    build_conversation_service,
    chat_avatar,
    chat_bubble_html,
    current_view,
    greets_instead_of_replying,
    history_for_display,
    init_session,
    llm_status_message,
    mask_sensitive_text,
    quick_actions,
    reset_conversation,
    start_chat,
    stored_welcome,
    submit_user_message,
    take_pending_message,
    typing_indicator_html,
)
from banco_agil.agents.state import ConversationState  # noqa: E402
from banco_agil.agents.triage import handle_triage  # noqa: E402
from banco_agil.config import Settings  # noqa: E402
from banco_agil.domain.enums import Agent, Intent  # noqa: E402
from banco_agil.domain.models import Client  # noqa: E402
from banco_agil.services.conversation import ConversationTurn  # noqa: E402
from banco_agil.services.welcome import DEFAULT_WELCOME_MESSAGE  # noqa: E402


@dataclass
class FakeConversationService:
    """Registra turnos e devolve respostas configuradas sem regra bancaria."""

    replies: list[str]
    calls: list[str]
    fail_with: BaseException | None = None
    pending_cpf: str | None = None

    def handle_turn(
        self,
        state: ConversationState,
        history: object,
        user_text: str,
    ) -> ConversationTurn:
        """Simula um turno completo do grafo."""
        del history
        self.calls.append(user_text)
        if self.fail_with is not None:
            raise self.fail_with
        if self.pending_cpf is not None:
            state.pending_cpf = self.pending_cpf
        reply = self.replies.pop(0) if self.replies else "resposta"
        updated_history = (HumanMessage(content=user_text), AIMessage(content=reply))
        return ConversationTurn(state=state, history=updated_history, reply=reply)


def test_init_session_preserves_existing_conversation() -> None:
    session: dict[str, object] = {}
    init_session(session)
    first_conversation = session["conversation"]

    init_session(session)

    assert session["conversation"] is first_conversation
    # A tela inicial faz a acolhida: o chat abre sem nenhuma fala previa.
    assert cast(list[BaseMessage], session["history"]) == []
    assert stored_welcome(session) == DEFAULT_WELCOME_MESSAGE
    assert session["notice"] is None


def test_reset_conversation_keeps_persistence_files(tmp_path: Path) -> None:
    persistence = tmp_path / "solicitacoes_aumento_limite.csv"
    persistence.write_text("conteudo", encoding="utf-8")
    session: dict[str, object] = {}
    init_session(session)
    session["history"] = [HumanMessage(content="oi")]

    reset_conversation(session)

    assert isinstance(session["conversation"], ConversationState)
    assert cast(list[BaseMessage], session["history"]) == []
    assert stored_welcome(session) == DEFAULT_WELCOME_MESSAGE
    assert session["notice"] is None
    assert persistence.read_text(encoding="utf-8") == "conteudo"


def test_session_uses_generated_welcome_message() -> None:
    session: dict[str, object] = {}

    init_session(session, "Boas-vindas geradas pelo modelo.")

    assert stored_welcome(session) == "Boas-vindas geradas pelo modelo."
    assert cast(list[BaseMessage], session["history"]) == []


def test_first_turn_puts_the_client_first_and_answers_with_the_welcome() -> None:
    session: dict[str, object] = {}
    init_session(session)
    service = FakeConversationService(replies=["tudo bem?"], calls=[], fail_with=None)

    reply = submit_user_message(
        session, cast(app.ConversationServiceLike, service), "  olá  "
    )

    # O texto exato segue para o servico; so a fala exibida vira a saudacao.
    assert service.calls == ["olá"]
    assert reply == DEFAULT_WELCOME_MESSAGE
    assert session["notice"] is None
    history = cast(list[BaseMessage], session["history"])
    assert isinstance(history[0], HumanMessage)
    assert [message.content for message in history] == [
        "olá",
        DEFAULT_WELCOME_MESSAGE,
    ]


def test_later_turns_return_the_service_reply() -> None:
    session: dict[str, object] = {}
    init_session(session)
    service = FakeConversationService(
        replies=["saudacao substituida", "tudo bem?"], calls=[], fail_with=None
    )
    submit_user_message(session, cast(app.ConversationServiceLike, service), "olá")

    reply = submit_user_message(
        session, cast(app.ConversationServiceLike, service), "e aí"
    )

    assert reply == "tudo bem?"
    assert service.calls == ["olá", "e aí"]


def test_first_turn_keeps_the_reply_when_the_cpf_was_recognized() -> None:
    session: dict[str, object] = {}
    init_session(session)
    service = FakeConversationService(
        replies=["CPF localizado."],
        calls=[],
        fail_with=None,
        pending_cpf="11144477735",
    )

    reply = submit_user_message(
        session, cast(app.ConversationServiceLike, service), "11144477735"
    )

    # Repetir a saudacao aqui pediria o CPF de novo, logo apos recebe-lo.
    assert reply == "CPF localizado."


def test_greeting_only_replaces_the_reply_while_nothing_advanced() -> None:
    fresh = ConversationState()
    with_cpf = ConversationState(pending_cpf="11144477735")
    failed = ConversationState(authentication_attempts=1)

    assert greets_instead_of_replying(fresh) is True
    assert greets_instead_of_replying(with_cpf) is False
    assert greets_instead_of_replying(failed) is False


def test_submit_empty_message_does_not_call_service() -> None:
    session: dict[str, object] = {}
    init_session(session)
    service = FakeConversationService(replies=[], calls=[], fail_with=None)

    reply = submit_user_message(
        session, cast(app.ConversationServiceLike, service), "   "
    )

    assert service.calls == []
    assert "Digite" in reply
    assert session["notice"] == reply
    assert cast(list[BaseMessage], session["history"]) == []


def test_submit_ended_conversation_returns_restart_guidance() -> None:
    from banco_agil.domain.exceptions import DomainError

    session: dict[str, object] = {}
    init_session(session)
    service = FakeConversationService(
        replies=[], calls=[], fail_with=DomainError("conversation is already ended")
    )

    reply = submit_user_message(
        session, cast(app.ConversationServiceLike, service), "oi"
    )

    assert "CPF" in reply
    assert "Reinicie" not in reply
    assert cast(list[BaseMessage], session["history"]) == []


def test_submit_cpf_after_end_starts_new_attendance() -> None:
    from banco_agil.domain.enums import EndReason

    session: dict[str, object] = {}
    init_session(session)
    ended_state = cast(ConversationState, session["conversation"])
    ended_state.end(EndReason.USER_REQUEST)
    service = FakeConversationService(
        replies=["CPF localizado."],
        calls=[],
        fail_with=None,
        pending_cpf="11144477735",
    )

    reply = submit_user_message(
        session, cast(app.ConversationServiceLike, service), "11144477735"
    )

    assert service.calls == ["11144477735"]
    assert reply == "CPF localizado."
    history = cast(list[BaseMessage], session["history"])
    assert [message.content for message in history] == [
        "11144477735",
        "CPF localizado.",
    ]


def test_submit_hides_technical_details_on_integration_failure() -> None:
    from banco_agil.domain.exceptions import RepositoryError

    session: dict[str, object] = {}
    init_session(session)
    service = FakeConversationService(
        replies=[],
        calls=[],
        fail_with=RepositoryError("clients CSV could not be read: /tmp/x"),
    )

    reply = submit_user_message(
        session, cast(app.ConversationServiceLike, service), "oi"
    )

    assert "Tente novamente" in reply
    assert "/tmp/x" not in reply
    assert "Traceback" not in reply
    assert cast(list[BaseMessage], session["history"]) == []


def test_mask_sensitive_text_hides_cpf_and_birth_date() -> None:
    masked = mask_sensitive_text("meu CPF 012.345.678-90 nasceu em 20/05/1990")

    assert "012.345.678-90" not in masked
    assert "20/05/1990" not in masked
    assert "***" in masked


def test_mask_sensitive_text_also_hides_internal_iso_date() -> None:
    assert "1990-05-20" not in mask_sensitive_text("nascimento 1990-05-20")


def test_mask_sensitive_text_hides_plain_cpf() -> None:
    assert "01234567890" not in mask_sensitive_text("cpf 01234567890")


def test_history_for_display_never_exposes_raw_cpf() -> None:
    history = [
        HumanMessage(content="meu cpf é 01234567890"),
        AIMessage(content="Dados confirmados."),
    ]

    displayed = history_for_display(history)

    assert displayed[0][0] == "user"
    assert displayed[1][0] == "assistant"
    assert "01234567890" not in displayed[0][1]


def test_chat_avatars_use_person_and_bank_emojis() -> None:
    assert chat_avatar("user") == "🧑"
    assert chat_avatar("assistant") == "🏦"
    assert chat_avatar("unknown") == "💬"


def test_chat_bubbles_identify_roles_and_escape_content() -> None:
    user_bubble = chat_bubble_html("user", "Olá <script>alert(1)</script>")
    assistant_bubble = chat_bubble_html("assistant", "Linha 1\nLinha 2")

    assert "chat-bubble--user" in user_bubble
    assert "chat-row--user" in user_bubble
    assert "🧑" in user_bubble
    assert "<script>" not in user_bubble
    assert "&lt;script&gt;" in user_bubble
    assert "chat-bubble--assistant" in assistant_bubble
    assert "chat-row--assistant" in assistant_bubble
    assert "🏦" in assistant_bubble
    assert "Linha 1<br>Linha 2" in assistant_bubble
    assert "chat-bubble__sender" not in user_bubble + assistant_bubble


def test_typing_indicator_uses_bank_and_three_animated_dots() -> None:
    indicator = typing_indicator_html()

    assert "🏦" in indicator
    assert indicator.count('class="typing-indicator__dot"') == 3
    assert 'aria-label="Digitando"' in indicator
    assert "Digitando..." not in indicator


def test_build_service_without_key_supports_deterministic_turns(
    tmp_path: Path,
) -> None:
    settings = Settings(
        groq_api_key=None,
        data_dir=tmp_path,
        awesomeapi_base_url="https://example.invalid",
    )

    service = build_conversation_service(settings)

    state = ConversationState()
    turn = service.handle_turn(state, (), "oi")

    assert "CPF" in turn.reply


def test_llm_status_reports_deterministic_mode_without_key() -> None:
    settings = Settings(_env_file=None, groq_api_key=None)

    message = llm_status_message(settings)

    assert "determinístico" in message
    assert "BANCO_AGIL_GROQ_API_KEY" in message


def test_llm_status_reports_configured_groq_model() -> None:
    settings = Settings(
        _env_file=None,
        groq_api_key="test-key",
        groq_model="test-model",
    )

    message = llm_status_message(settings)

    assert "Groq ativo" in message
    assert "boas-vindas" in message
    assert "especialistas" in message
    assert "test-model" in message
    assert "test-key" not in message


def test_build_service_uses_fictitious_client_csv(tmp_path: Path) -> None:
    (tmp_path / "clientes.csv").write_text(
        "cpf,data_nascimento,limite_credito,score_credito\n"
        "01234567890,1990-05-20,2500.00,700\n",
        encoding="utf-8",
    )
    settings = Settings(groq_api_key=None, data_dir=tmp_path)
    service = build_conversation_service(settings)
    state = ConversationState()
    service.handle_turn(state, (), "01234567890")
    turn = service.handle_turn(state, (), "20/05/1990")

    client = state.authenticated_client
    assert client is not None
    assert isinstance(client, Client)
    assert client.birth_date == date(1990, 5, 20)
    assert client.credit_limit == Decimal("2500.00")
    assert "ajudar" in turn.reply.casefold()


@dataclass
class UnusedAuthenticationService:
    """Falha se a triagem tentar autenticar um cliente ja autenticado."""

    def validate_cpf(self, state: ConversationState, cpf: str) -> object:
        """Nunca deve ser chamada com o cliente ja autenticado."""
        raise AssertionError("triagem nao deveria revalidar CPF")

    def authenticate(self, state: ConversationState, birth_date: str) -> object:
        """Nunca deve ser chamada com o cliente ja autenticado."""
        raise AssertionError("triagem nao deveria reautenticar")


def _authenticated_client() -> Client:
    return Client(
        cpf="01234567890",
        birth_date=date(1990, 5, 20),
        credit_limit=Decimal("2500.00"),
        credit_score=700,
    )


def test_session_starts_on_landing_view() -> None:
    session: dict[str, object] = {}

    init_session(session)

    assert current_view(session) == LANDING_VIEW
    assert take_pending_message(session) is None


def test_current_view_ignores_unknown_value() -> None:
    assert current_view({"view": "galeria"}) == LANDING_VIEW


def test_start_chat_schedules_message_and_opens_chat() -> None:
    session: dict[str, object] = {}
    init_session(session)

    assert start_chat(session, "  Quero ver a cotação de moedas.  ") is True

    assert current_view(session) == CHAT_VIEW
    assert take_pending_message(session) == "Quero ver a cotação de moedas."


def test_start_chat_ignores_blank_text_and_keeps_landing() -> None:
    session: dict[str, object] = {}
    init_session(session)

    assert start_chat(session, "   ") is False

    assert current_view(session) == LANDING_VIEW
    assert take_pending_message(session) is None


def test_pending_message_is_delivered_only_once() -> None:
    session: dict[str, object] = {}
    init_session(session)
    start_chat(session, "Quero visualizar meu limite de crédito.")

    first = take_pending_message(session)
    second = take_pending_message(session)

    assert first == "Quero visualizar meu limite de crédito."
    assert second is None


def test_reset_conversation_returns_to_landing_and_drops_pending() -> None:
    session: dict[str, object] = {}
    init_session(session)
    start_chat(session, "Quero ver a cotação de moedas.")

    reset_conversation(session)

    assert current_view(session) == LANDING_VIEW
    assert take_pending_message(session) is None


def test_quick_actions_offer_the_four_services() -> None:
    actions = quick_actions()

    # O rotulo e texto de vitrine e pode mudar; a chave identifica o servico.
    assert [action.key for action in actions] == [
        "credit_limit",
        "limit_increase",
        "credit_interview",
        "exchange_rate",
    ]
    assert all(isinstance(action, QuickAction) for action in actions)
    assert all(action.label.strip() for action in actions)
    assert all(action.icon.strip() for action in actions)
    assert all(action.prompt.strip() for action in actions)
    assert len({action.label for action in actions}) == len(actions)


@pytest.mark.parametrize(
    ("action_key", "expected_intent", "expected_agent"),
    (
        ("credit_limit", Intent.CREDIT_LIMIT, Agent.CREDIT),
        ("limit_increase", Intent.LIMIT_INCREASE, Agent.CREDIT),
        ("credit_interview", Intent.CREDIT_INTERVIEW, Agent.CREDIT_INTERVIEW),
        ("exchange_rate", Intent.EXCHANGE_RATE, Agent.EXCHANGE),
    ),
)
def test_quick_action_prompt_routes_without_llm(
    action_key: str,
    expected_intent: Intent,
    expected_agent: Agent,
) -> None:
    action = next(item for item in quick_actions() if item.key == action_key)
    state = ConversationState(authenticated_client=_authenticated_client())

    reply = handle_triage(
        state,
        action.prompt,
        UnusedAuthenticationService(),
        llm=None,
        turn_id="quick-action-turn",
    )

    assert reply
    assert state.intent is expected_intent
    assert state.active_agent is expected_agent
    assert state.pending_flow is None
