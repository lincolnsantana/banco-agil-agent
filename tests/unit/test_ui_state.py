"""Testes do estado e das acoes da interface Streamlit."""

import sys
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage  # noqa: E402

import app  # noqa: E402
from app import (  # noqa: E402
    build_conversation_service,
    end_conversation,
    history_for_display,
    init_session,
    llm_status_message,
    mask_sensitive_text,
    reset_conversation,
    submit_user_message,
)
from banco_agil.agents.state import ConversationState
from banco_agil.config import Settings
from banco_agil.domain.models import Client
from banco_agil.services.conversation import ConversationTurn
from banco_agil.services.welcome import DEFAULT_WELCOME_MESSAGE


@dataclass
class FakeConversationService:
    """Registra turnos e devolve respostas configuradas sem regra bancaria."""

    replies: list[str]
    calls: list[str]
    fail_with: BaseException | None = None

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
        reply = self.replies.pop(0) if self.replies else "resposta"
        updated_history = (HumanMessage(content=user_text), AIMessage(content=reply))
        return ConversationTurn(state=state, history=updated_history, reply=reply)


def test_init_session_preserves_existing_conversation() -> None:
    session: dict[str, object] = {}
    init_session(session)
    first_conversation = session["conversation"]

    init_session(session)

    assert session["conversation"] is first_conversation
    history = cast(list[BaseMessage], session["history"])
    assert len(history) == 1
    assert isinstance(history[0], AIMessage)
    assert history[0].content == DEFAULT_WELCOME_MESSAGE
    assert session["notice"] is None


def test_reset_conversation_keeps_persistence_files(tmp_path: Path) -> None:
    persistence = tmp_path / "solicitacoes_aumento_limite.csv"
    persistence.write_text("conteudo", encoding="utf-8")
    session: dict[str, object] = {}
    init_session(session)
    session["history"] = [HumanMessage(content="oi")]

    reset_conversation(session)

    assert isinstance(session["conversation"], ConversationState)
    history = cast(list[BaseMessage], session["history"])
    assert len(history) == 1
    assert isinstance(history[0], AIMessage)
    assert history[0].content == DEFAULT_WELCOME_MESSAGE
    assert session["notice"] is None
    assert persistence.read_text(encoding="utf-8") == "conteudo"


def test_session_uses_generated_welcome_message() -> None:
    session: dict[str, object] = {}

    init_session(session, "Boas-vindas geradas pelo modelo.")

    history = cast(list[BaseMessage], session["history"])
    assert history[0].content == "Boas-vindas geradas pelo modelo."


def test_submit_forwards_exact_text_and_updates_session() -> None:
    session: dict[str, object] = {}
    init_session(session)
    service = FakeConversationService(replies=["tudo bem?"], calls=[], fail_with=None)

    reply = submit_user_message(
        session, cast(app.ConversationServiceLike, service), "  olá  "
    )

    assert reply == "tudo bem?"
    assert service.calls == ["olá"]
    assert session["notice"] is None
    history = cast(list[BaseMessage], session["history"])
    assert [message.content for message in history] == ["olá", "tudo bem?"]


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
    assert len(cast(list[BaseMessage], session["history"])) == 1


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

    assert "Reinicie" in reply
    assert len(cast(list[BaseMessage], session["history"])) == 1


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
    assert len(cast(list[BaseMessage], session["history"])) == 1


def test_end_conversation_routes_through_service() -> None:
    session: dict[str, object] = {}
    init_session(session)
    service = FakeConversationService(
        replies=["Atendimento encerrado."], calls=[], fail_with=None
    )

    reply = end_conversation(session, cast(app.ConversationServiceLike, service))

    assert service.calls == ["encerrar"]
    assert "encerrado" in reply.casefold()


def test_mask_sensitive_text_hides_cpf_and_birth_date() -> None:
    masked = mask_sensitive_text("meu CPF 012.345.678-90 nasceu em 1990-05-20")

    assert "012.345.678-90" not in masked
    assert "1990-05-20" not in masked
    assert "***" in masked


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
    turn = service.handle_turn(state, (), "1990-05-20")

    client = state.authenticated_client
    assert client is not None
    assert isinstance(client, Client)
    assert client.birth_date == date(1990, 5, 20)
    assert client.credit_limit == Decimal("2500.00")
    assert "ajudar" in turn.reply.casefold()
