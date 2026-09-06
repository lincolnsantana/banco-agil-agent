"""Testes do estado e das acoes da interface Streamlit."""

import re
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
    chat_bubble_html,
    current_view,
    greets_instead_of_replying,
    history_for_display,
    init_session,
    mask_sensitive_text,
    quick_actions,
    remember_suggestions,
    reset_conversation,
    return_to_landing,
    start_chat,
    stored_suggestions,
    stored_welcome,
    submit_user_message,
    suggestions_for,
    take_pending_message,
    typing_hold_seconds,
    typing_indicator_html,
)
from banco_agil.agents.state import ConversationState  # noqa: E402
from banco_agil.agents.triage import handle_triage  # noqa: E402
from banco_agil.config import Settings  # noqa: E402
from banco_agil.domain.enums import Agent, EndReason, Intent  # noqa: E402
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
    responding_agent: Agent | None = None

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
        return ConversationTurn(
            state=state,
            history=updated_history,
            reply=reply,
            responding_agent=self.responding_agent,
        )


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


def test_submit_ended_conversation_points_to_the_new_service_button() -> None:
    from banco_agil.domain.exceptions import DomainError

    session: dict[str, object] = {}
    init_session(session)
    service = FakeConversationService(
        replies=[], calls=[], fail_with=DomainError("conversation is already ended")
    )

    reply = submit_user_message(
        session, cast(app.ConversationServiceLike, service), "oi"
    )

    # O caminho oferecido agora e o botao; digitar CPF continua funcionando,
    # mas deixou de ser a instrucao.
    assert "novo atendimento" in reply.casefold()
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


def test_brand_sits_in_the_native_header_on_both_screens() -> None:
    from urllib.parse import unquote

    import app as ui

    # Desenhada em pseudo-elemento do header: nao entra na arvore do Streamlit,
    # entao vale nas duas telas sem nenhum elemento por tela.
    assert '[data-testid="stHeader"]::before' in ui._UI_STYLES
    assert 'content: "Banco Ágil"' in ui._UI_STYLES
    assert "__BRAND_MARK__" not in ui._UI_STYLES
    # O simbolo viaja embutido: sem arquivo estatico para servir.
    assert ui._BRAND_MARK_URI.startswith("data:image/svg+xml,")
    # As cores precisam do escape: um # cru encerraria a url() do CSS.
    assert "#" not in ui._BRAND_MARK_URI
    assert unquote(ui._BRAND_MARK_URI).endswith("</svg>")


def test_hero_heading_lives_only_on_the_landing_screen() -> None:
    import app as ui

    # A acolhida abre a tela inicial; o chat abre limpo, sem repetir cabeçalho.
    # Os textos são vitrine e mudam: o contrato é cada elemento existir.
    assert 'class="landing-hero__brand"' in ui._LANDING_HERO
    assert 'class="landing-hero__subtitle"' in ui._LANDING_HERO
    assert not hasattr(ui, "_PAGE_HEADING")


def test_landing_hero_is_emitted_as_a_single_html_line() -> None:
    import app as ui

    # Quebra dentro do bloco ja colou palavras e confundiu o markdown.
    assert "\n" not in ui._LANDING_HERO.strip()


def test_landing_hero_uses_only_tags_streamlit_leaves_intact() -> None:
    import re

    import app as ui

    # O react-markdown do Streamlit troca h1..h6 por componentes proprios e a
    # classe do autor se perde. div e span atravessam, como nos balões.
    tags = set(re.findall(r"<(\w+)", ui._LANDING_HERO))
    assert tags <= {"div", "span"}
    # A semantica de titulo continua, via ARIA.
    assert 'role="heading"' in ui._LANDING_HERO


def test_chat_lines_identify_roles_and_escape_content() -> None:
    user_line = chat_bubble_html("user", "Olá <script>alert(1)</script>")
    assistant_line = chat_bubble_html("assistant", "Linha 1\nLinha 2")

    # So o cliente tem balao; o especialista escreve direto na pagina.
    assert "chat-bubble--user" in user_line
    assert "chat-row--user" in user_line
    assert "<script>" not in user_line
    assert "&lt;script&gt;" in user_line
    assert "chat-text--assistant" in assistant_line
    assert "chat-row--assistant" in assistant_line
    assert "chat-bubble" not in assistant_line
    assert "Linha 1<br>Linha 2" in assistant_line
    assert "chat-bubble__sender" not in user_line + assistant_line


def test_chat_lines_carry_no_avatar_emoji() -> None:
    rendered = "".join(
        (
            chat_bubble_html("user", "oi"),
            chat_bubble_html("assistant", "olá"),
            typing_indicator_html(),
        )
    )

    assert not re.search(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", rendered)


def test_typing_dots_stay_visible_even_when_the_turn_is_instant() -> None:
    # Os fluxos deterministicos respondem em milissegundos; sem o piso, a
    # pergunta e a resposta apareceriam no mesmo instante.
    assert typing_hold_seconds(0.0) == pytest.approx(app._TYPING_MIN_SECONDS)
    assert typing_hold_seconds(app._TYPING_MIN_SECONDS / 2) == pytest.approx(
        app._TYPING_MIN_SECONDS / 2
    )


def test_typing_dots_do_not_delay_a_turn_that_already_took_long() -> None:
    assert typing_hold_seconds(app._TYPING_MIN_SECONDS) == 0.0
    assert typing_hold_seconds(12.0) == 0.0


def test_typing_indicator_shows_three_animated_dots() -> None:
    indicator = typing_indicator_html()

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


def _finished_service_state() -> ConversationState:
    """Estado de um servico concluido: autenticado e sem passo em andamento."""
    return ConversationState(authenticated_client=_authenticated_client())


@pytest.mark.parametrize(
    "responding_agent",
    (
        Agent.TRIAGE,
        Agent.CREDIT,
        Agent.CREDIT_INTERVIEW,
        Agent.EXCHANGE,
        Agent.KNOWLEDGE,
    ),
)
def test_finished_service_offers_three_questions_for_each_agent(
    responding_agent: Agent,
) -> None:
    suggestions = suggestions_for(_finished_service_state(), responding_agent)

    assert len(suggestions) == 3
    assert all(isinstance(item, QuickAction) for item in suggestions)
    assert all(item.label.strip() and item.prompt.strip() for item in suggestions)
    assert len({item.key for item in suggestions}) == 3


def test_suggestions_fall_back_to_triage_without_responding_agent() -> None:
    assert suggestions_for(_finished_service_state(), None) == suggestions_for(
        _finished_service_state(), Agent.TRIAGE
    )


def test_specialist_holding_the_turn_offers_no_questions() -> None:
    # O credito ainda espera o valor do aumento: sugerir outro assunto agora
    # atrapalharia a coleta.
    state = ConversationState(
        authenticated_client=_authenticated_client(),
        active_agent=Agent.CREDIT,
        intent=Intent.LIMIT_INCREASE,
    )

    assert suggestions_for(state, Agent.CREDIT) == ()


def test_pending_offer_awaiting_yes_or_no_offers_no_questions() -> None:
    state = ConversationState(
        authenticated_client=_authenticated_client(),
        pending_flow=Intent.CREDIT_INTERVIEW,
    )

    assert suggestions_for(state, Agent.KNOWLEDGE) == ()


def test_unauthenticated_or_ended_conversation_offers_no_questions() -> None:
    ended = ConversationState(authenticated_client=_authenticated_client())
    ended.end(EndReason.USER_REQUEST)

    assert suggestions_for(ConversationState(), Agent.TRIAGE) == ()
    assert suggestions_for(ended, Agent.TRIAGE) == ()


def test_turn_stores_the_questions_of_the_agent_that_answered() -> None:
    session: dict[str, object] = {}
    init_session(session)
    session["conversation"] = _finished_service_state()
    service = FakeConversationService(
        replies=["Seu limite atual é R$ 2.500,00."],
        calls=[],
        responding_agent=Agent.CREDIT,
    )

    submit_user_message(
        session, cast(app.ConversationServiceLike, service), "qual meu limite?"
    )

    assert stored_suggestions(session) == suggestions_for(
        _finished_service_state(), Agent.CREDIT
    )


def test_failed_turn_leaves_no_questions_behind() -> None:
    session: dict[str, object] = {}
    init_session(session)
    session["conversation"] = _finished_service_state()
    remember_suggestions(
        session, suggestions_for(_finished_service_state(), Agent.TRIAGE)
    )
    service = FakeConversationService(
        replies=[], calls=[], fail_with=RuntimeError("indisponivel")
    )

    submit_user_message(
        session, cast(app.ConversationServiceLike, service), "qual meu limite?"
    )

    assert stored_suggestions(session) == ()


def test_stored_suggestions_ignores_unexpected_value() -> None:
    assert stored_suggestions({"suggestions": "credito"}) == ()


@pytest.mark.parametrize(
    ("suggestion_key", "expected_intent", "expected_agent"),
    (
        ("credit_limit", Intent.CREDIT_LIMIT, Agent.CREDIT),
        ("limit_increase", Intent.LIMIT_INCREASE, Agent.CREDIT),
        ("credit_interview", Intent.CREDIT_INTERVIEW, Agent.CREDIT_INTERVIEW),
        ("exchange_dollar", Intent.EXCHANGE_RATE, Agent.EXCHANGE),
        ("exchange_euro", Intent.EXCHANGE_RATE, Agent.EXCHANGE),
        ("score_rule", Intent.INFORMATION, Agent.KNOWLEDGE),
        ("quote_source", Intent.INFORMATION, Agent.KNOWLEDGE),
    ),
)
def test_suggestion_prompt_routes_without_llm(
    suggestion_key: str,
    expected_intent: Intent,
    expected_agent: Agent,
) -> None:
    suggestion = next(
        item
        for group in app._AGENT_SUGGESTIONS.values()
        for item in group
        if item.key == suggestion_key
    )
    state = ConversationState(authenticated_client=_authenticated_client())

    reply = handle_triage(
        state,
        suggestion.prompt,
        UnusedAuthenticationService(),
        llm=None,
        turn_id="suggestion-turn",
    )

    assert reply
    assert state.intent is expected_intent
    assert state.active_agent is expected_agent


def test_stored_suggestions_survive_a_rerun_of_the_script() -> None:
    """O rerun redefine `QuickAction`; a sessao guarda chave, nao objeto."""
    session: dict[str, object] = {}
    init_session(session)
    suggestions = suggestions_for(_finished_service_state(), Agent.EXCHANGE)

    remember_suggestions(session, suggestions)

    assert all(isinstance(item, str) for item in cast(tuple, session["suggestions"]))
    assert stored_suggestions(session) == suggestions


def test_chat_placeholder_invites_writing_next_to_the_suggestions() -> None:
    session: dict[str, object] = {}
    init_session(session)

    assert app._chat_placeholder(session) == "Digite sua mensagem"

    remember_suggestions(
        session, suggestions_for(_finished_service_state(), Agent.TRIAGE)
    )

    assert app._chat_placeholder(session) == "Ou pergunte outra coisa..."


def test_back_to_landing_keeps_the_conversation_alive() -> None:
    session: dict[str, object] = {}
    init_session(session)
    session["conversation"] = _finished_service_state()
    session["history"] = [HumanMessage(content="oi"), AIMessage(content="olá")]
    remember_suggestions(
        session, suggestions_for(_finished_service_state(), Agent.TRIAGE)
    )
    start_chat(session, "qual meu limite?")

    return_to_landing(session)

    assert current_view(session) == LANDING_VIEW
    # A mensagem agendada nao pode disparar sozinha depois da volta.
    assert take_pending_message(session) is None
    assert stored_suggestions(session) == ()
    assert len(cast(list[BaseMessage], session["history"])) == 2
    assert cast(ConversationState, session["conversation"]).authenticated


def test_new_service_after_the_end_starts_a_clean_conversation() -> None:
    session: dict[str, object] = {}
    init_session(session)
    ended = cast(ConversationState, session["conversation"])
    ended.end(EndReason.USER_REQUEST)
    session["history"] = [HumanMessage(content="tchau"), AIMessage(content="até logo")]
    remember_suggestions(
        session, suggestions_for(_finished_service_state(), Agent.TRIAGE)
    )
    session["view"] = CHAT_VIEW

    # O que o botao de novo atendimento executa.
    reset_conversation(session, stored_welcome(session))

    conversation = cast(ConversationState, session["conversation"])
    assert not conversation.ended
    assert conversation.end_reason is None
    assert cast(list[BaseMessage], session["history"]) == []
    assert current_view(session) == LANDING_VIEW
    assert stored_suggestions(session) == ()


def test_ended_notice_confirms_the_end_without_asking_for_a_cpf() -> None:
    assert "encerrado" in app.ENDED_NOTICE.casefold()
    assert "cpf" not in app.ENDED_NOTICE.casefold()
