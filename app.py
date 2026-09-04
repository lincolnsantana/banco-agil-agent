"""Interface Streamlit do atendimento bancario conversacional."""

import re
from collections.abc import MutableMapping, Sequence
from html import escape
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
from banco_agil.services.welcome import (
    DEFAULT_WELCOME_MESSAGE,
    generate_welcome_message,
)

_CONVERSATION_KEY = "conversation"
_HISTORY_KEY = "history"
_NOTICE_KEY = "notice"

_CHAT_AVATARS = {"assistant": "🏦", "user": "🧑"}

_UI_STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --agil-surface: var(--background-color);
    --agil-surface-muted: var(--secondary-background-color);
    --agil-text: var(--text-color);
    --agil-accent: var(--primary-color);
    --agil-border: color-mix(in srgb, var(--text-color) 18%, transparent);
    --agil-muted: color-mix(in srgb, var(--text-color) 68%, transparent);
    --agil-shadow: color-mix(in srgb, var(--text-color) 10%, transparent);
    --agil-assistant-bubble: #374151;
    --agil-assistant-border: #64748b;
    --agil-user-bubble: #0b5cad;
    --agil-user-border: #60a5fa;
}

html,
body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] * {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stHeader"],
[data-testid="stBottom"] {
    background: var(--agil-surface);
}

[data-testid="stMainBlockContainer"] {
    max-width: 880px;
    padding-top: 1.25rem;
    padding-bottom: 7rem;
}

.page-heading {
    margin: 0 0 1.5rem;
    padding: 0.25rem 0;
}

.page-heading__title {
    margin: 0;
    color: var(--agil-text);
    font-size: clamp(1.55rem, 4vw, 2rem);
    font-weight: 700;
    letter-spacing: -0.035em;
    line-height: 1.15;
}

.page-heading__subtitle {
    margin: 0.35rem 0 0;
    color: var(--agil-muted);
    font-size: 0.9rem;
    font-weight: 500;
}

.chat-row {
    display: flex;
    width: 100%;
    align-items: flex-start;
    gap: 0.65rem;
    margin: 0.75rem 0;
}

.chat-row--user {
    flex-direction: row-reverse;
}

.chat-avatar {
    display: grid;
    flex: 0 0 38px;
    width: 38px;
    height: 38px;
    place-items: center;
    border: 1px solid var(--agil-border);
    border-radius: 50%;
    background: var(--agil-surface-muted);
    font-size: 1.15rem;
    line-height: 1;
    box-shadow: 0 2px 8px var(--agil-shadow);
}

.chat-row--user .chat-avatar {
    border-color: color-mix(in srgb, var(--agil-accent) 48%, var(--agil-border));
    background: color-mix(in srgb, var(--agil-accent) 14%, var(--agil-surface));
}

.chat-bubble {
    position: relative;
    width: fit-content;
    max-width: min(76%, 650px);
    padding: 0.85rem 1.05rem;
    border: 2px solid var(--agil-border);
    border-radius: 17px;
    color: #ffffff;
    font-size: 0.94rem;
    line-height: 1.55;
    overflow-wrap: anywhere;
    box-shadow: 0 5px 18px var(--agil-shadow);
}

.chat-bubble--assistant {
    margin-right: auto;
    border-color: var(--agil-assistant-border);
    border-top-left-radius: 5px;
    background: var(--agil-assistant-bubble);
}

.chat-bubble--user {
    margin-left: auto;
    border-color: var(--agil-user-border);
    border-top-right-radius: 5px;
    background: var(--agil-user-bubble);
}

[data-testid="stChatInput"] {
    border-color: var(--agil-border);
    border-radius: 16px;
    color: var(--agil-text);
    background: var(--agil-surface);
    box-shadow: 0 8px 24px var(--agil-shadow);
}

[data-testid="stChatInput"] textarea {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

[data-testid="stChatInput"]:focus-within,
[data-testid="stChatInput"] [data-baseweb="textarea"]:focus-within {
    border-color: var(--agil-user-bubble) !important;
    box-shadow: 0 0 0 1px var(--agil-user-bubble),
        0 8px 24px var(--agil-shadow) !important;
}

[data-testid="stChatInput"] textarea:focus {
    outline: none !important;
    box-shadow: none !important;
    caret-color: var(--agil-user-bubble);
}

.typing-indicator {
    display: inline-flex;
    align-items: center;
    gap: 0.32rem;
    min-height: 42px;
    padding: 0.65rem 0.9rem;
    border: 2px solid var(--agil-assistant-border);
    border-radius: 17px;
    background: var(--agil-assistant-bubble);
    box-shadow: 0 5px 18px var(--agil-shadow);
}

.typing-indicator__dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #ffffff;
    animation: typing-bounce 1.15s infinite ease-in-out;
}

.typing-indicator__dot:nth-child(2) {
    animation-delay: 0.15s;
}

.typing-indicator__dot:nth-child(3) {
    animation-delay: 0.3s;
}

@keyframes typing-bounce {
    0%, 60%, 100% {
        opacity: 0.45;
        transform: translateY(0);
    }

    30% {
        opacity: 1;
        transform: translateY(-4px);
    }
}

@media (max-width: 640px) {
    [data-testid="stMainBlockContainer"] {
        padding: 0.75rem 0.8rem 6.5rem;
    }

    .page-heading {
        margin-bottom: 1rem;
    }

    .chat-bubble {
        max-width: calc(100% - 52px);
    }

    .chat-avatar {
        flex-basis: 34px;
        width: 34px;
        height: 34px;
    }
}
</style>
"""

_PAGE_HEADING = """
<header class="page-heading">
    <h1 class="page-heading__title">🏦 Banco Ágil: Atendimento Digital</h1>
    <p class="page-heading__subtitle">Cuide do seu crédito de forma simples: consulte seu limite, peça aumento, faça sua análise <br> e acompanhe cotações de moedas.
</p>
</header>
"""

_TYPING_INDICATOR = """
<div class="chat-row chat-row--assistant" role="status" aria-label="Digitando">
    <span class="chat-avatar" aria-hidden="true">🏦</span>
    <span class="typing-indicator" aria-hidden="true">
        <span class="typing-indicator__dot"></span>
        <span class="typing-indicator__dot"></span>
        <span class="typing-indicator__dot"></span>
    </span>
</div>
"""

_CPF_FORMATTED_PATTERN = re.compile(r"\d{3}\.\d{3}\.\d{3}-\d{2}")
_CPF_PLAIN_PATTERN = re.compile(r"\b\d{11}\b")
_ISO_BIRTH_DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_BR_BIRTH_DATE_PATTERN = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")


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


def init_session(
    session: MutableMapping[str, object],
    welcome_message: str = DEFAULT_WELCOME_MESSAGE,
) -> None:
    """Garante conversa, historico e aviso sem descartar o existente."""
    if _CONVERSATION_KEY not in session:
        session[_CONVERSATION_KEY] = ConversationState()
    if _HISTORY_KEY not in session:
        session[_HISTORY_KEY] = [AIMessage(content=welcome_message)]
    if _NOTICE_KEY not in session:
        session[_NOTICE_KEY] = None


def reset_conversation(
    session: MutableMapping[str, object],
    welcome_message: str = DEFAULT_WELCOME_MESSAGE,
) -> None:
    """Reinicia a conversa em memoria sem apagar nenhuma persistencia."""
    session[_CONVERSATION_KEY] = ConversationState()
    session[_HISTORY_KEY] = [AIMessage(content=welcome_message)]
    session[_NOTICE_KEY] = None


def mask_sensitive_text(text: str) -> str:
    """Oculta CPF e data de nascimento antes de exibir na interface."""
    masked = _CPF_FORMATTED_PATTERN.sub("***", text)
    masked = _CPF_PLAIN_PATTERN.sub("***", masked)
    masked = _ISO_BIRTH_DATE_PATTERN.sub("***", masked)
    return _BR_BIRTH_DATE_PATTERN.sub("***", masked)


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


def chat_avatar(role: str) -> str:
    """Retorna o avatar visual seguro para cada participante do chat."""
    return _CHAT_AVATARS.get(role, "💬")


def chat_bubble_html(role: str, text: str) -> str:
    """Monta linha e balão escapados para impedir HTML vindo da conversa."""
    bubble_role = "user" if role == "user" else "assistant"
    avatar = escape(chat_avatar(bubble_role))
    safe_text = escape(text).replace("\n", "<br>")
    return (
        f'<div class="chat-row chat-row--{bubble_role}">'
        f'<span class="chat-avatar" aria-hidden="true">{avatar}</span>'
        f'<div class="chat-bubble chat-bubble--{bubble_role}">'
        f"{safe_text}</div></div>"
    )


def typing_indicator_html() -> str:
    """Retorna o indicador acessível de resposta em andamento."""
    return _TYPING_INDICATOR


def render_chat_message(role: str, text: str) -> None:
    """Renderiza uma mensagem no balão e avatar correspondentes."""
    st.markdown(chat_bubble_html(role, text), unsafe_allow_html=True)


def build_conversation_service(settings: Settings) -> ConversationService:
    """Compõe o serviço com repositórios, câmbio e LLM opcional.

    O LLM é usado somente quando há chave configurada; sem chave, a apresentação
    e as respostas dos especialistas usam seus textos canônicos.
    """
    return _build_conversation_service(settings, _optional_llm(settings))


def _build_conversation_service(
    settings: Settings,
    llm: GroqStructuredLlm | None,
) -> ConversationService:
    client_repository = ClientCsvRepository(settings.data_dir / "clientes.csv")
    dependencies = GraphDependencies(
        authentication=AuthenticationService(client_repository),
        credit=CreditService(
            ScoreLimitCsvRepository(settings.data_dir / "score_limite.csv"),
            CreditRequestCsvRepository(
                settings.data_dir / "solicitacoes_aumento_limite.csv"
            ),
            client_repository,
        ),
        credit_interview=CreditInterviewService(client_repository),
        exchange=ExchangeService(AwesomeApiClient(settings.awesomeapi_base_url)),
        llm=llm,
    )
    return ConversationService(build_graph(dependencies))


def llm_status_message(settings: Settings) -> str:
    """Descreve o modo conversacional sem expor a credencial configurada."""
    api_key = settings.groq_api_key
    if api_key is not None and api_key.get_secret_value().strip():
        model = settings.groq_model
        return f"Groq ativo nas boas-vindas e nos especialistas — modelo {model}."
    return (
        "Modo determinístico: Groq inativo. Configure "
        "BANCO_AGIL_GROQ_API_KEY no arquivo .env e reinicie a aplicação."
    )


def _looks_like_cpf(text: str) -> bool:
    """Indica se o texto contém 11 dígitos para reabrir o atendimento."""
    return len(re.sub(r"\D", "", text)) == 11


def submit_user_message(
    session: MutableMapping[str, object],
    service: ConversationServiceLike,
    user_text: str,
    welcome_message: str = DEFAULT_WELCOME_MESSAGE,
) -> str:
    """Encaminha a entrada ao serviço e atualiza a sessão.

    Erros recuperáveis viram mensagens amigáveis, sem detalhe interno, e não
    corrompem a conversa em andamento. Com o atendimento encerrado, um CPF
    inicia outro automaticamente.
    """
    init_session(session, welcome_message)
    text = user_text.strip()
    if not text:
        notice = "Digite uma mensagem para continuar."
        session[_NOTICE_KEY] = notice
        return notice
    state = cast(ConversationState, session[_CONVERSATION_KEY])
    if state.ended and _looks_like_cpf(text):
        reset_conversation(session, welcome_message)
        state = cast(ConversationState, session[_CONVERSATION_KEY])
    history = cast(Sequence[BaseMessage], session[_HISTORY_KEY])
    try:
        turn = service.handle_turn(state, history, text)
    except DomainError:
        notice = (
            "Este atendimento foi encerrado. "
            "Para começar outro, informe seu CPF com 11 dígitos."
        )
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
def _get_runtime() -> tuple[ConversationService, GroqStructuredLlm | None]:
    settings = Settings()
    llm = _optional_llm(settings)
    return _build_conversation_service(settings, llm), llm


def main() -> None:
    """Renderiza o chat e encaminha cada entrada ao serviço de conversa."""
    st.set_page_config(page_title="Banco Ágil - Atendimento", page_icon="🏦")
    st.markdown(_UI_STYLES, unsafe_allow_html=True)
    st.markdown(_PAGE_HEADING, unsafe_allow_html=True)
    settings = Settings()
    st.caption(llm_status_message(settings))
    session = cast(MutableMapping[str, object], st.session_state)
    service, llm = _get_runtime()
    if _HISTORY_KEY not in session:
        init_session(session, generate_welcome_message(llm))
    else:
        init_session(session)

    state = cast(ConversationState, session[_CONVERSATION_KEY])
    history = cast(Sequence[BaseMessage], session[_HISTORY_KEY])
    for role, safe_text in history_for_display(history):
        render_chat_message(role, safe_text)

    notice = session[_NOTICE_KEY]
    if isinstance(notice, str) and notice:
        st.warning(notice)
    if state.ended:
        st.info(
            "Atendimento encerrado. "
            "Para um novo atendimento, informe seu CPF com 11 dígitos."
        )

    user_input = st.chat_input("Digite sua mensagem")
    if user_input is not None:
        render_chat_message("user", mask_sensitive_text(user_input.strip()))
        typing_placeholder = st.empty()
        typing_placeholder.markdown(typing_indicator_html(), unsafe_allow_html=True)
        try:
            submit_user_message(
                session, service, user_input, generate_welcome_message(llm)
            )
        finally:
            typing_placeholder.empty()
        st.rerun()


if __name__ == "__main__" and st.runtime.exists():
    main()
