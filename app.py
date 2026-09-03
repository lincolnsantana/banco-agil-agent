"""Interface Streamlit do atendimento bancario conversacional."""

import re
from collections.abc import MutableMapping, Sequence
from typing import Protocol, cast

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from banco_agil.agents.graph import GraphDependencies, build_graph
from banco_agil.agents.state import ConversationState
from banco_agil.config import Settings
from banco_agil.domain.exceptions import DomainError
from banco_agil.integrations.awesomeapi import AwesomeApiClient
from banco_agil.integrations.llm import GroqStructuredLlm
from banco_agil.repositories.client_csv import ClientCsvRepository
from banco_agil.repositories.credit_request_csv import CreditRequestCsvRepository
from banco_agil.repositories.score_limit_csv import ScoreLimitCsvRepository
from banco_agil.services.authentication import AuthenticationService
from banco_agil.services.conversation import ConversationService, ConversationTurn
from banco_agil.services.credit import CreditService
from banco_agil.services.credit_interview import CreditInterviewService
from banco_agil.services.exchange import ExchangeService

_CONVERSATION_KEY = "conversation"
_HISTORY_KEY = "history"
_NOTICE_KEY = "notice"

_CPF_FORMATTED_PATTERN = re.compile(r"\d{3}\.\d{3}\.\d{3}-\d{2}")
_CPF_PLAIN_PATTERN = re.compile(r"\b\d{11}\b")
_BIRTH_DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


class ConversationServiceLike(Protocol):
    """Contrato estrutural usado pela UI para executar um turno."""

    def handle_turn(
        self,
        state: ConversationState,
        history: Sequence[BaseMessage],
        user_text: str,
    ) -> ConversationTurn:
        """Executa um turno e retorna estado, historico e resposta."""
        ...


def init_session(session: MutableMapping[str, object]) -> None:
    """Garante conversa, historico e aviso sem descartar o existente."""
    if _CONVERSATION_KEY not in session:
        session[_CONVERSATION_KEY] = ConversationState()
    if _HISTORY_KEY not in session:
        session[_HISTORY_KEY] = []
    if _NOTICE_KEY not in session:
        session[_NOTICE_KEY] = None


def reset_conversation(session: MutableMapping[str, object]) -> None:
    """Reinicia a conversa em memoria sem apagar nenhuma persistencia."""
    session[_CONVERSATION_KEY] = ConversationState()
    session[_HISTORY_KEY] = []
    session[_NOTICE_KEY] = None


def mask_sensitive_text(text: str) -> str:
    """Oculta CPF e data de nascimento antes de exibir na interface."""
    masked = _CPF_FORMATTED_PATTERN.sub("***", text)
    masked = _CPF_PLAIN_PATTERN.sub("***", masked)
    return _BIRTH_DATE_PATTERN.sub("***", masked)


def history_for_display(history: Sequence[BaseMessage]) -> list[tuple[str, str]]:
    """Converte o historico em pares (papel, texto) seguros para exibicao."""
    displayed: list[tuple[str, str]] = []
    for message in history:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue
        if not isinstance(message.content, str):
            continue
        displayed.append((role, mask_sensitive_text(message.content)))
    return displayed


def build_conversation_service(settings: Settings) -> ConversationService:
    """Compõe o serviço com repositórios, câmbio e LLM opcional.

    O LLM é usado somente quando há chave configurada; sem chave, os fluxos
    determinísticos continuam funcionando e a intenção ambígua pede
    esclarecimento.
    """
    client_repository = ClientCsvRepository(settings.data_dir / "clientes.csv")
    dependencies = GraphDependencies(
        authentication=AuthenticationService(client_repository),
        credit=CreditService(
            ScoreLimitCsvRepository(settings.data_dir / "score_limite.csv"),
            CreditRequestCsvRepository(
                settings.data_dir / "solicitacoes_aumento_limite.csv"
            ),
        ),
        credit_interview=CreditInterviewService(client_repository),
        exchange=ExchangeService(AwesomeApiClient(settings.awesomeapi_base_url)),
        llm=_optional_llm(settings),
    )
    return ConversationService(build_graph(dependencies))


def llm_status_message(settings: Settings) -> str:
    """Descreve o modo conversacional sem expor a credencial configurada."""
    api_key = settings.groq_api_key
    if api_key is not None and api_key.get_secret_value().strip():
        model = settings.groq_model
        return f"Groq ativo para humanização e intenções ambíguas — modelo {model}."
    return (
        "Modo determinístico: Groq inativo. Configure "
        "BANCO_AGIL_GROQ_API_KEY no arquivo .env e reinicie a aplicação."
    )


def submit_user_message(
    session: MutableMapping[str, object],
    service: ConversationServiceLike,
    user_text: str,
) -> str:
    """Encaminha a entrada ao serviço e atualiza a sessão.

    Erros recuperáveis viram mensagens amigáveis, sem detalhe interno, e não
    corrompem a conversa em andamento.
    """
    init_session(session)
    text = user_text.strip()
    if not text:
        notice = "Digite uma mensagem para continuar."
        session[_NOTICE_KEY] = notice
        return notice
    state = cast(ConversationState, session[_CONVERSATION_KEY])
    history = cast(Sequence[BaseMessage], session[_HISTORY_KEY])
    try:
        turn = service.handle_turn(state, history, text)
    except DomainError:
        notice = "Este atendimento já foi encerrado. Reinicie para começar outro."
        session[_NOTICE_KEY] = notice
        return notice
    except Exception:
        notice = "Não foi possível processar sua mensagem agora. Tente novamente."
        session[_NOTICE_KEY] = notice
        return notice
    session[_CONVERSATION_KEY] = turn.state
    session[_HISTORY_KEY] = list(turn.history)
    session[_NOTICE_KEY] = None
    return turn.reply


def end_conversation(
    session: MutableMapping[str, object],
    service: ConversationServiceLike,
) -> str:
    """Encerra o atendimento pelo mesmo caminho de uma mensagem do cliente."""
    return submit_user_message(session, service, "encerrar")


def _optional_llm(settings: Settings) -> GroqStructuredLlm | None:
    """Cria o LLM somente quando há chave configurada, sem exigir rede."""
    api_key = settings.groq_api_key
    if api_key is None or not api_key.get_secret_value().strip():
        return None
    try:
        return GroqStructuredLlm(settings)
    except ValueError:
        return None


@st.cache_resource
def _get_service() -> ConversationService:
    return build_conversation_service(Settings())


def main() -> None:
    """Renderiza o chat e encaminha cada entrada ao serviço de conversa."""
    st.set_page_config(page_title="Banco Ágil", page_icon="🏦")
    st.title("Banco Ágil — Atendimento")
    settings = Settings()
    st.caption(llm_status_message(settings))
    session = cast(MutableMapping[str, object], st.session_state)
    init_session(session)
    service = _get_service()

    state = cast(ConversationState, session[_CONVERSATION_KEY])
    history = cast(Sequence[BaseMessage], session[_HISTORY_KEY])
    for role, safe_text in history_for_display(history):
        st.chat_message(role).write(safe_text)

    notice = session[_NOTICE_KEY]
    if isinstance(notice, str) and notice:
        st.warning(notice)
    if state.ended:
        st.info("Atendimento encerrado. Reinicie para iniciar um novo atendimento.")

    end_clicked, restart_clicked = st.columns(2)
    if end_clicked.button("Encerrar atendimento"):
        end_conversation(session, service)
        st.rerun()
    if restart_clicked.button("Reiniciar atendimento"):
        reset_conversation(session)
        st.rerun()

    user_input = st.chat_input("Digite sua mensagem")
    if user_input is not None:
        with st.spinner("Processando..."):
            submit_user_message(session, service, user_input)
        st.rerun()


if __name__ == "__main__" and st.runtime.exists():
    main()
