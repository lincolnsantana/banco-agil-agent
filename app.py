"""Interface Streamlit do atendimento bancario conversacional."""

import re
import time
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
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
_WELCOME_KEY = "welcome_message"
_VIEW_KEY = "view"
_PENDING_KEY = "pending_message"

LANDING_VIEW = "landing"
CHAT_VIEW = "chat"

# Tempo da animacao de saida da tela inicial antes de trocar para o chat.
_TRANSITION_SECONDS = 0.28
_QUICK_ACTION_COLUMNS = 4

_CHAT_AVATARS = {"assistant": "🏦", "user": "🧑"}


@dataclass(frozen=True)
class QuickAction:
    """Atalho da tela inicial que vira mensagem do cliente ao ser clicado."""

    key: str
    label: str
    icon: str
    prompt: str


# Cada prompt usa termos que a triagem deterministica ja reconhece, para que o
# atalho chegue ao especialista correto sem depender do LLM.
_QUICK_ACTIONS = (
    QuickAction(
        key="credit_limit",
        label="Visualizar limite",
        icon="💳",
        prompt="Quero visualizar meu limite de crédito.",
    ),
    QuickAction(
        key="limit_increase",
        label="Aumento de crédito",
        icon="📈",
        prompt="Quero solicitar um aumento do meu limite de crédito.",
    ),
    QuickAction(
        key="credit_interview",
        label="Atualizar score",
        icon="📝",
        prompt="Quero fazer a entrevista de crédito para atualizar meu limite.",
    ),
    QuickAction(
        key="exchange_rate",
        label="Cotação de moedas",
        icon="💱",
        prompt="Quero ver a cotação de moedas.",
    ),
)

_UI_STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    /* O Streamlit nao expoe variaveis CSS do tema: var(--text-color) e
       companhia nao existem e invalidam a regra inteira. As cores neutras
       saem de currentColor, que segue theme.textColor do config.toml
       (branco no modo escuro, preto no claro); as de marca sao fixas. */
    --agil-accent: #0b5cad;
    --agil-assistant-bubble: #374151;
    --agil-assistant-border: #64748b;
    --agil-user-bubble: #0b5cad;
    --agil-user-border: #60a5fa;
    --agil-ease: cubic-bezier(0.16, 1, 0.3, 1);
    --agil-control-radius: 999px;
    --agil-field-radius: 26px;
    --agil-border: color-mix(in srgb, currentColor 22%, transparent);
    --agil-muted: color-mix(in srgb, currentColor 65%, transparent);
    /* Sombra fixa: currentColor deixaria um brilho branco no modo escuro. */
    --agil-shadow: rgba(0, 0, 0, 0.28);
    --agil-tint: color-mix(in srgb, currentColor 8%, transparent);
}

html,
body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] * {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
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
    background: var(--agil-tint);
    font-size: 1.15rem;
    line-height: 1;
    box-shadow: 0 2px 8px var(--agil-shadow);
}

.chat-row--user .chat-avatar {
    border-color: color-mix(in srgb, var(--agil-accent) 48%, var(--agil-border));
    background: color-mix(in srgb, var(--agil-accent) 14%, transparent);
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

/* O elemento com data-testid="stChatInput" e apenas um wrapper de
   posicionamento: quem desenha borda, fundo e raio e o div filho direto.
   Estilizar o wrapper nao produz efeito visivel algum. */
[data-testid="stChatInput"] > div {
    border-radius: var(--agil-field-radius) !important;
    box-shadow: 0 8px 24px var(--agil-shadow);
}

[data-testid="stChatInput"] textarea,
[data-testid="stChatInputTextArea"] {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: transparent;
}

[data-testid="stChatInput"] > div:focus-within {
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

/* ----------------------------- Tela inicial ----------------------------- */

.st-key-landing {
    min-height: 74vh;
    justify-content: center;
    animation: agil-view-in 520ms var(--agil-ease) both;
}

.landing-hero {
    margin: 0 auto 1.8rem;
    text-align: center;
    animation: agil-rise 560ms var(--agil-ease) both;
}

.landing-hero__title {
    margin: 0;
    font-size: clamp(1.8rem, 5vw, 2.6rem);
    font-weight: 400;
    letter-spacing: -0.04em;
    line-height: 1.12;
}

.st-key-landing_input {
    animation: agil-rise 560ms var(--agil-ease) 90ms both;
}

.st-key-quick_actions {
    margin-top: 0.9rem;
    animation: agil-rise 560ms var(--agil-ease) 170ms both;
}

.st-key-quick_actions button {
    min-height: 46px;
    border-radius: var(--agil-control-radius) !important;
    font-size: 0.86rem;
    font-weight: 500;
    transition: transform 160ms ease;
}

.st-key-quick_actions button:hover {
    transform: translateY(-1px);
}

.st-key-quick_actions button:focus-visible {
    outline: none;
    border-color: var(--agil-user-bubble);
    box-shadow: 0 0 0 2px color-mix(in srgb, var(--agil-accent) 45%, transparent);
}

/* ------------------------------- Tela chat ------------------------------ */

.st-key-chat_view {
    animation: agil-view-in 460ms var(--agil-ease) both;
}

.st-key-restart_chat button {
    border-radius: var(--agil-control-radius) !important;
    font-size: 0.8rem;
    font-weight: 500;
}

@keyframes agil-view-in {
    from {
        opacity: 0;
        transform: translateY(16px);
    }

    to {
        opacity: 1;
        transform: none;
    }
}

@keyframes agil-rise {
    from {
        opacity: 0;
        transform: translateY(18px);
    }

    to {
        opacity: 1;
        transform: none;
    }
}

@keyframes agil-view-out {
    to {
        opacity: 0;
        transform: translateY(-14px) scale(0.985);
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

    .st-key-landing {
        min-height: 68vh;
    }

    .st-key-quick_actions button {
        font-size: 0.8rem;
    }
}

@media (prefers-reduced-motion: reduce) {
    .st-key-landing,
    .st-key-landing_input,
    .st-key-quick_actions,
    .st-key-chat_view,
    .landing-hero,
    .typing-indicator__dot {
        animation: none !important;
    }
}
</style>
"""

# Injetado apos a tela inicial ja estar na pagina: reaproveita o elemento
# existente para animar a saida sem redesenhar os widgets.
_LANDING_EXIT_STYLE = """
<style>
.st-key-landing {
    animation: agil-view-out 260ms cubic-bezier(0.4, 0, 1, 1) forwards !important;
}
</style>
"""

_LANDING_HERO = """
<section class="landing-hero">
    <h1 class="landing-hero__title">Como posso ajudar você hoje?</h1>
</section>
"""

_PAGE_HEADING = """
<header class="page-heading">
    <h1 class="page-heading__title">🏦 Banco Ágil: Atendimento Digital</h1>
    <p class="page-heading__subtitle">
        Cuide do seu crédito de forma simples: consulte seu limite, peça
        aumento, faça sua análise e acompanhe cotações de moedas.
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
    """Garante conversa, historico, aviso e tela sem descartar o existente."""
    if _CONVERSATION_KEY not in session:
        session[_CONVERSATION_KEY] = ConversationState()
    if _HISTORY_KEY not in session:
        session[_HISTORY_KEY] = []
    if _WELCOME_KEY not in session:
        session[_WELCOME_KEY] = welcome_message
    if _NOTICE_KEY not in session:
        session[_NOTICE_KEY] = None
    if _VIEW_KEY not in session:
        session[_VIEW_KEY] = LANDING_VIEW


def reset_conversation(
    session: MutableMapping[str, object],
    welcome_message: str = DEFAULT_WELCOME_MESSAGE,
) -> None:
    """Reinicia a conversa em memoria sem apagar nenhuma persistencia."""
    session[_CONVERSATION_KEY] = ConversationState()
    session[_HISTORY_KEY] = []
    session[_WELCOME_KEY] = welcome_message
    session[_NOTICE_KEY] = None
    session[_VIEW_KEY] = LANDING_VIEW
    session.pop(_PENDING_KEY, None)


def stored_welcome(
    session: MutableMapping[str, object],
    default: str = DEFAULT_WELCOME_MESSAGE,
) -> str:
    """Devolve a saudacao ja fixada na sessao, sem pedir outra ao modelo."""
    welcome = session.get(_WELCOME_KEY)
    return welcome if isinstance(welcome, str) and welcome.strip() else default


def quick_actions() -> tuple[QuickAction, ...]:
    """Retorna os atalhos oferecidos na tela inicial, em ordem de exibicao."""
    return _QUICK_ACTIONS


def current_view(session: MutableMapping[str, object]) -> str:
    """Informa a tela ativa, assumindo a inicial enquanto nada foi escolhido."""
    view = session.get(_VIEW_KEY)
    return view if view in {LANDING_VIEW, CHAT_VIEW} else LANDING_VIEW


def start_chat(session: MutableMapping[str, object], user_text: str) -> bool:
    """Agenda a primeira mensagem e abre o chat; texto vazio nao troca a tela."""
    text = user_text.strip()
    if not text:
        return False
    session[_PENDING_KEY] = text
    session[_VIEW_KEY] = CHAT_VIEW
    return True


def take_pending_message(session: MutableMapping[str, object]) -> str | None:
    """Consome a mensagem agendada para que ela seja enviada uma unica vez."""
    pending = session.pop(_PENDING_KEY, None)
    if isinstance(pending, str) and pending.strip():
        return pending
    return None


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


def greets_instead_of_replying(state: ConversationState) -> bool:
    """Indica se a saudacao substitui a resposta da primeira troca.

    So vale quando o turno nao avancou a autenticacao nem encerrou: se o
    cliente ja mandou um CPF valido ou errou a tentativa, a resposta real do
    atendimento e a util e a saudacao seria confusa.
    """
    return (
        not state.authenticated
        and not state.ended
        and state.pending_cpf is None
        and state.authentication_attempts == 0
    )


def _history_with_reply(
    history: Sequence[BaseMessage],
    reply: str,
) -> list[BaseMessage]:
    """Troca o texto da ultima fala do assistente pelo texto informado."""
    updated = list(history)
    for index in range(len(updated) - 1, -1, -1):
        if isinstance(updated[index], AIMessage):
            updated[index] = AIMessage(content=reply)
            return updated
    updated.append(AIMessage(content=reply))
    return updated


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
        session[_VIEW_KEY] = CHAT_VIEW
        state = cast(ConversationState, session[_CONVERSATION_KEY])
    history = cast(Sequence[BaseMessage], session[_HISTORY_KEY])
    first_turn = not history
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
    session[_NOTICE_KEY] = None
    if first_turn and greets_instead_of_replying(turn.state):
        greeting = stored_welcome(session, welcome_message)
        session[_HISTORY_KEY] = _history_with_reply(turn.history, greeting)
        return greeting
    session[_HISTORY_KEY] = list(turn.history)
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


def _render_quick_actions() -> str | None:
    """Desenha os atalhos em grade e devolve a mensagem do atalho clicado."""
    selected: str | None = None
    actions = quick_actions()
    for start in range(0, len(actions), _QUICK_ACTION_COLUMNS):
        row = actions[start : start + _QUICK_ACTION_COLUMNS]
        columns = st.columns(_QUICK_ACTION_COLUMNS, gap="small")
        for index, action in enumerate(row):
            with columns[index]:
                clicked = st.button(
                    action.label,
                    key=f"quick_action_{action.key}",
                    icon=action.icon,
                    use_container_width=True,
                )
            if clicked:
                selected = action.prompt
    return selected


def _render_landing(session: MutableMapping[str, object]) -> None:
    """Mostra a apresentação inicial e abre o chat na primeira interação."""
    with st.container(key="landing"):
        st.markdown(_LANDING_HERO, unsafe_allow_html=True)
        # Dentro de um container o chat_input fica na propria coluna, e nao
        # ancorado ao rodape, o que mantem o campo centralizado na abertura.
        with st.container(key="landing_input"):
            typed = st.chat_input(
                "Descreva o que você precisa",
                key="landing_chat_input",
            )
        with st.container(key="quick_actions"):
            chosen = _render_quick_actions()

    requested = chosen or (typed if isinstance(typed, str) else "")
    if start_chat(session, requested):
        st.markdown(_LANDING_EXIT_STYLE, unsafe_allow_html=True)
        time.sleep(_TRANSITION_SECONDS)
        st.rerun()


def _process_message(
    session: MutableMapping[str, object],
    service: ConversationServiceLike,
    user_text: str,
) -> None:
    """Exibe a mensagem do cliente, sinaliza digitação e executa o turno."""
    render_chat_message("user", mask_sensitive_text(user_text.strip()))
    typing_placeholder = st.empty()
    typing_placeholder.markdown(typing_indicator_html(), unsafe_allow_html=True)
    try:
        submit_user_message(session, service, user_text, stored_welcome(session))
    finally:
        typing_placeholder.empty()


def _render_chat(
    session: MutableMapping[str, object],
    settings: Settings,
    service: ConversationServiceLike,
    llm: GroqStructuredLlm | None,
) -> None:
    """Mostra o histórico, processa a entrada e oferece novo atendimento."""
    user_input = st.chat_input("Digite sua mensagem")
    with st.container(key="chat_view"):
        heading_column, restart_column = st.columns(
            [5, 2], gap="small", vertical_alignment="center"
        )
        with heading_column:
            st.markdown(_PAGE_HEADING, unsafe_allow_html=True)
        with restart_column, st.container(key="restart_chat"):
            restart = st.button(
                "Novo atendimento",
                key="restart_chat_button",
                icon="🔄",
                use_container_width=True,
                help="Volta para a tela inicial e começa outro atendimento.",
            )
        if restart:
            reset_conversation(session, generate_welcome_message(llm))
            st.rerun()

        st.caption(llm_status_message(settings))
        history = cast(Sequence[BaseMessage], session[_HISTORY_KEY])
        for role, safe_text in history_for_display(history):
            render_chat_message(role, safe_text)

        pending = take_pending_message(session)
        message = pending if pending is not None else user_input
        if isinstance(message, str) and message.strip():
            _process_message(session, service, message)
            st.rerun()

        notice = session[_NOTICE_KEY]
        if isinstance(notice, str) and notice:
            st.warning(notice)
        state = cast(ConversationState, session[_CONVERSATION_KEY])
        if state.ended:
            st.info(
                "Atendimento encerrado. "
                "Para um novo atendimento, informe seu CPF com 11 dígitos."
            )


def main() -> None:
    """Escolhe entre a tela inicial e o chat conforme o estado da sessão."""
    st.set_page_config(page_title="Banco Ágil - Atendimento", page_icon="🏦")
    st.markdown(_UI_STYLES, unsafe_allow_html=True)
    settings = Settings()
    session = cast(MutableMapping[str, object], st.session_state)
    service, llm = _get_runtime()
    if _HISTORY_KEY not in session:
        init_session(session, generate_welcome_message(llm))
    else:
        init_session(session)

    if current_view(session) == LANDING_VIEW:
        _render_landing(session)
        return
    _render_chat(session, settings, service, llm)


if __name__ == "__main__" and st.runtime.exists():
    main()
