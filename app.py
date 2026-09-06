"""Interface Streamlit do atendimento bancario conversacional."""

import re
import time
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from html import escape
from typing import Protocol, cast
from urllib.parse import quote

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import SecretStr
from streamlit.delta_generator import DeltaGenerator

from banco_agil.agents.graph import GraphDependencies, build_graph
from banco_agil.agents.state import ConversationState
from banco_agil.config import Settings
from banco_agil.domain.enums import Agent
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
from banco_agil.services.knowledge import KnowledgeService
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
_SUGGESTIONS_KEY = "suggestions"

LANDING_VIEW = "landing"
CHAT_VIEW = "chat"

# Tempo da animacao de saida da tela inicial antes de trocar para o chat.
_TRANSITION_SECONDS = 0.28
# Tempo minimo de exibicao dos pontos de digitacao. Os fluxos deterministicos
# respondem em milissegundos, e sem esse piso a pergunta do cliente e a resposta
# aparecem no mesmo instante, como se ninguem tivesse digitado nada.
_TYPING_MIN_SECONDS = 0.7
_QUICK_ACTION_COLUMNS = 4
_SUGGESTION_ROWS = 3


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

# Perguntas rapidas oferecidas dentro do chat quando um especialista conclui o
# atendimento. Cada trio continua a conversa pelo assunto vizinho ao que acabou
# de ser respondido, e os textos usam os mesmos termos que a triagem reconhece
# sem LLM, para o clique chegar ao especialista certo.
_SUGGESTION_LIMIT = QuickAction(
    key="credit_limit",
    label="Ver meu limite",
    icon="💳",
    prompt="Quero visualizar meu limite de crédito.",
)
_SUGGESTION_INCREASE = QuickAction(
    key="limit_increase",
    label="Pedir aumento",
    icon="📈",
    prompt="Quero solicitar um aumento do meu limite de crédito.",
)
_SUGGESTION_INTERVIEW = QuickAction(
    key="credit_interview",
    label="Atualizar meu score",
    icon="📝",
    prompt="Quero fazer a entrevista de crédito para atualizar meu limite.",
)
_SUGGESTION_DOLLAR = QuickAction(
    key="exchange_dollar",
    label="Cotação do dólar",
    icon="💱",
    prompt="Quero ver a cotação do dólar.",
)
_SUGGESTION_EURO = QuickAction(
    key="exchange_euro",
    label="Cotação do euro",
    icon="💶",
    prompt="Quero ver a cotação do euro.",
)
_SUGGESTION_SCORE_RULE = QuickAction(
    key="score_rule",
    label="Como é calculado o score?",
    icon="❓",
    prompt="Como é calculado o meu score?",
)
_SUGGESTION_QUOTE_SOURCE = QuickAction(
    key="quote_source",
    label="De onde vem a cotação?",
    icon="❓",
    prompt="De onde vem a cotação que vocês mostram?",
)

_AGENT_SUGGESTIONS: dict[Agent, tuple[QuickAction, ...]] = {
    Agent.TRIAGE: (
        _SUGGESTION_LIMIT,
        _SUGGESTION_INCREASE,
        _SUGGESTION_DOLLAR,
    ),
    Agent.CREDIT: (
        _SUGGESTION_INCREASE,
        _SUGGESTION_INTERVIEW,
        _SUGGESTION_DOLLAR,
    ),
    Agent.CREDIT_INTERVIEW: (
        _SUGGESTION_LIMIT,
        _SUGGESTION_INCREASE,
        _SUGGESTION_SCORE_RULE,
    ),
    Agent.EXCHANGE: (
        _SUGGESTION_EURO,
        _SUGGESTION_QUOTE_SOURCE,
        _SUGGESTION_LIMIT,
    ),
    Agent.KNOWLEDGE: (
        _SUGGESTION_LIMIT,
        _SUGGESTION_INTERVIEW,
        _SUGGESTION_DOLLAR,
    ),
}

_SUGGESTION_BY_KEY = {
    suggestion.key: suggestion
    for group in _AGENT_SUGGESTIONS.values()
    for suggestion in group
}

# A marca desenhada no proprio CSS: um data URI dispensa servir arquivo estatico
# (o Streamlit so entrega o que esta em `static/`, desabilitado por padrao) e nao
# depende de <svg> sobreviver a sanitizacao do markdown.
_BRAND_MARK_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 512 512'>"
    "<defs><linearGradient id='agil' x1='0' y1='0' x2='1' y2='1'>"
    "<stop offset='0' stop-color='#4F46E5'/>"
    "<stop offset='.5' stop-color='#A855F7'/>"
    "<stop offset='1' stop-color='#FF5C7A'/>"
    "</linearGradient></defs>"
    "<circle cx='256' cy='256' r='256' fill='url(#agil)'/>"
    "<path d='M128 384 L256 165 L384 384' fill='none' stroke='#FFFFFF' "
    "stroke-width='66' stroke-linecap='round' stroke-linejoin='round'/>"
    "</svg>"
)
_BRAND_MARK_URI = "data:image/svg+xml," + quote(_BRAND_MARK_SVG)

_UI_STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    /* O Streamlit nao expoe variaveis CSS do tema: var(--text-color) e
       companhia nao existem e invalidam a regra inteira. As cores neutras
       saem de currentColor, que segue theme.textColor do config.toml
       (branco no modo escuro, preto no claro); as de marca sao fixas. */
    --agil-accent: #0b5cad;
    --agil-user-bubble: #6d28d9;
    --agil-ease: cubic-bezier(0.16, 1, 0.3, 1);
    --agil-control-radius: 999px;
    --agil-field-radius: 26px;
    --agil-border: color-mix(in srgb, currentColor 22%, transparent);
    --agil-muted: color-mix(in srgb, currentColor 65%, transparent);
    /* Sombra fixa: currentColor deixaria um brilho branco no modo escuro. */
    --agil-shadow: rgba(0, 0, 0, 0.28);
    --agil-tint: color-mix(in srgb, currentColor 8%, transparent);
    --agil-control-fill: color-mix(in srgb, currentColor 13%, transparent);
    --agil-control-fill-hover: color-mix(in srgb, currentColor 20%, transparent);
}

/* A marca mora no header nativo, que ja e fixo no topo: um pseudo-elemento nao
   entra na arvore do Streamlit, nao disputa espaco com o conteudo e acompanha
   as duas telas. O texto herda a cor do tema; o simbolo vem do data URI. */
[data-testid="stHeader"]::before {
    position: fixed;
    top: 0.8rem;
    left: 1.25rem;
    display: flex;
    align-items: center;
    height: 50px;
    /* Abre a esquerda para o simbolo, que vem como fundo: recuo, 36px de
       largura e o respiro ate o nome. */
    padding: 0 1.25rem 0 3.4rem;
    border-radius: var(--agil-control-radius);
    background:
        url("__BRAND_MARK__") 0.5rem center / 36px 36px no-repeat,
        var(--agil-control-fill);
    content: "Banco Ágil";
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: -0.01em;
}

html,
body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] * {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* O header do Streamlit e absolute em top 0 com 3.75rem de altura, e o proprio
   framework reserva 6rem de topo por causa disso. Reduzir esse espaco joga o
   conteudo por baixo do header. */
[data-testid="stMainBlockContainer"] {
    max-width: 880px;
    padding-top: 6.5rem;
    padding-bottom: 7rem;
}

/* Quem fala se le pelo alinhamento, sem avatar dos dois lados: o especialista
   escreve direto na pagina, como um documento, e so a fala do cliente ganha
   balao. */
.chat-row {
    display: flex;
    width: 100%;
    margin: 1.35rem 0;
}

.chat-row--user {
    justify-content: flex-end;
}

.chat-text {
    max-width: 100%;
    color: inherit;
    font-size: 0.94rem;
    line-height: 1.65;
    overflow-wrap: anywhere;
}

.chat-bubble {
    width: fit-content;
    max-width: min(76%, 650px);
    padding: 0.7rem 1.05rem;
    border-radius: 20px;
    background: var(--agil-user-bubble);
    color: #ffffff;
    font-size: 0.94rem;
    line-height: 1.55;
    overflow-wrap: anywhere;
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

/* Sem balao para o especialista, os pontos batem na cor do texto da pagina. */
.typing-indicator {
    display: inline-flex;
    align-items: center;
    gap: 0.32rem;
    min-height: 24px;
    /* O atraso curto faz os pontos entrarem depois da fala do cliente, e nao
       junto com ela: a conversa ganha ordem em vez de surgir pronta. */
    animation: agil-rise 320ms var(--agil-ease) 120ms both;
}

.typing-indicator__dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: currentColor;
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

.landing-hero__brand {
    margin: 0;
    font-size: clamp(2rem, 5.5vw, 3rem);
    font-weight: 300;
    letter-spacing: -0.02em;
    line-height: 1.2;
}

.landing-hero__subtitle {
    margin: 0.6rem auto 0;
    max-width: 46rem;
    color: var(--agil-muted);
    font-size: 0.9rem;
    font-weight: 500;
    line-height: 1.5;
}

/* O bloco da tela inicial e um flex column com min-height para centralizar;
   sem travar a base, o campo absorve a sobra vertical e estica. O campo do
   chat nao passa por isso porque mora na barra inferior. */
.st-key-landing_input,
.st-key-landing_input [data-testid="stElementContainer"],
.st-key-landing_input [data-testid="stChatInput"],
.st-key-landing_input [data-testid="stChatInput"] > div {
    flex: 0 0 auto;
    height: auto;
}

.st-key-landing_input {
    animation: agil-rise 560ms var(--agil-ease) 90ms both;
}

.st-key-quick_actions {
    margin-top: 0.9rem;
    animation: agil-rise 560ms var(--agil-ease) 170ms both;
}

.st-key-quick_actions button,\n.st-key-new_service button {
    min-height: 54px;
    border: none !important;
    border-radius: var(--agil-control-radius) !important;
    background: var(--agil-control-fill) !important;
    font-size: 0.86rem;
    font-weight: 600;
    transition: transform 160ms ease, background 160ms ease;
}

/* O rotulo mora num <p> com peso proprio dentro do botao: sem alcancar esse
   elemento, font-weight no <button> nao muda nada. */
.st-key-quick_actions button p,\n.st-key-new_service button p {
    font-weight: 600 !important;
}

.st-key-quick_actions button:hover,\n.st-key-new_service button:hover {
    background: var(--agil-control-fill-hover) !important;
    transform: translateY(-1px);
}

.st-key-quick_actions button:focus-visible,\n.st-key-new_service button:focus-visible {
    outline: none;
    box-shadow: 0 0 0 2px color-mix(in srgb, var(--agil-accent) 55%, transparent);
}

/* ------------------------------- Tela chat ------------------------------ */

/* Fecha a conversa: o aviso de encerramento e o convite para recomecar ficam
   juntos, no fim do historico. */
.st-key-new_service {
    align-items: center;
    margin-top: 1.1rem;
}

.st-key-chat_view {
    animation: agil-view-in 460ms var(--agil-ease) both;
}

/* Fica logo abaixo do header do Streamlit (3.75rem) e na borda esquerda da
   coluna da conversa, acompanhando-a ate a tela estreitar. Preso na viewport:
   voltar continua a um clique depois de rolar a conversa. */
.st-key-back_to_landing {
    position: fixed;
    top: 4.35rem;
    left: max(1rem, calc(50% - 440px));
    z-index: 20;
    width: auto;
    animation: agil-view-in 460ms var(--agil-ease) both;
}

.st-key-back_to_landing button {
    min-height: 32px;
    padding: 0 0.85rem;
    border: 1px solid var(--agil-border) !important;
    border-radius: var(--agil-control-radius) !important;
    backdrop-filter: blur(8px);
    background: var(--agil-control-fill) !important;
    color: inherit;
    font-size: 0.8rem;
    font-weight: 500;
    transition: background 160ms ease;
}

.st-key-back_to_landing button:hover {
    background: var(--agil-control-fill-hover) !important;
}

.st-key-back_to_landing button:focus-visible {
    outline: none;
    box-shadow: 0 0 0 2px color-mix(in srgb, var(--agil-accent) 55%, transparent);
}

/* O painel de perguntas rapidas fica ancorado na barra inferior, logo acima do
   campo de texto, e acompanha a largura da conversa. */
[data-testid="stBottomBlockContainer"] {
    max-width: 880px;
    margin: 0 auto;
}

.st-key-chat_suggestions {
    overflow: hidden;
    margin-bottom: 0.6rem;
    border: 1px solid var(--agil-border);
    border-radius: 18px;
    background: var(--agil-tint);
    box-shadow: 0 10px 30px var(--agil-shadow);
    animation: agil-rise 420ms var(--agil-ease) both;
}

/* O Streamlit separa blocos e colunas com gap de 1rem e ainda folga os
   containers de elemento. Dentro do cartao esse respiro vira espaco morto
   acima do titulo e da primeira opcao, entao ele e zerado aqui. */
.st-key-chat_suggestions,
.st-key-chat_suggestions [data-testid="stVerticalBlock"],
.st-key-chat_suggestions [data-testid="stHorizontalBlock"],
.st-key-chat_suggestions [data-testid="stColumn"] {
    gap: 0 !important;
}

.st-key-chat_suggestions [data-testid="stElementContainer"],
.st-key-chat_suggestions [data-testid="stMarkdown"],
.st-key-chat_suggestions [data-testid="stMarkdownContainer"] {
    margin: 0 !important;
    padding: 0 !important;
}

.st-key-chat_suggestions_header {
    padding: 0.5rem 0.35rem 0.5rem 1rem;
    border-bottom: 1px solid var(--agil-border);
}

.suggestions-title {
    color: var(--agil-muted);
    font-size: 0.84rem;
    font-weight: 500;
    line-height: 1.5;
}

/* A numeracao vem do CSS para o rotulo do botao continuar sendo so a pergunta,
   que e o texto enviado ao atendimento. */
.st-key-chat_suggestion_rows {
    counter-reset: agil-suggestion;
    gap: 0 !important;
}

.st-key-chat_suggestion_rows
[data-testid="stElementContainer"]:not(:first-child) button {
    border-top: 1px solid var(--agil-border) !important;
}

.st-key-chat_suggestion_rows button {
    justify-content: flex-start !important;
    min-height: 46px;
    padding: 0 1rem;
    border: none !important;
    border-radius: 0 !important;
    background: transparent !important;
    color: inherit;
    font-size: 0.9rem;
    font-weight: 400;
    text-align: left;
    transition: background 160ms ease;
}

.st-key-chat_suggestion_rows button::before {
    display: grid;
    flex: 0 0 24px;
    width: 24px;
    height: 24px;
    margin-right: 0.75rem;
    place-items: center;
    border-radius: 7px;
    background: var(--agil-control-fill);
    color: var(--agil-muted);
    content: counter(agil-suggestion);
    counter-increment: agil-suggestion;
    font-size: 0.75rem;
}

.st-key-chat_suggestion_rows button:hover,
.st-key-chat_suggestion_rows button:focus-visible {
    outline: none;
    background: var(--agil-control-fill) !important;
}

.st-key-chat_suggestions_dismiss button {
    min-height: 34px;
    padding: 0;
    border: none !important;
    background: transparent !important;
    color: var(--agil-muted);
    font-size: 0.95rem;
}

.st-key-chat_suggestions_dismiss button:hover {
    background: var(--agil-control-fill) !important;
    color: inherit;
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
    [data-testid="stHeader"]::before {
        left: 0.8rem;
        height: 44px;
        padding: 0 1rem 0 3rem;
        background-size: 32px 32px;
        font-size: 0.82rem;
    }

    [data-testid="stMainBlockContainer"] {
        /* O header tem a mesma altura no celular: o topo continua reservado. */
        padding: 4.5rem 0.8rem 6.5rem;
    }

    .chat-bubble {
        max-width: 88%;
    }

    .st-key-landing {
        min-height: 68vh;
    }

    .st-key-quick_actions button {
        font-size: 0.8rem;
    }

    .st-key-chat_suggestion_rows button {
        font-size: 0.84rem;
    }

    .st-key-back_to_landing {
        top: 4.2rem;
        left: 0.8rem;
    }
}

@media (prefers-reduced-motion: reduce) {
    .st-key-landing,
    .st-key-landing_input,
    .st-key-quick_actions,
    .st-key-chat_suggestions,
    .st-key-back_to_landing,
    .st-key-chat_view,
    .landing-hero,
    .typing-indicator,
    .typing-indicator__dot {
        animation: none !important;
    }
}
</style>
""".replace("__BRAND_MARK__", _BRAND_MARK_URI)

# Injetado apos a tela inicial ja estar na pagina: reaproveita o elemento
# existente para animar a saida sem redesenhar os widgets.
_LANDING_EXIT_STYLE = """
<style>
.st-key-landing {
    animation: agil-view-out 260ms cubic-bezier(0.4, 0, 1, 1) forwards !important;
}
</style>
"""

_LANDING_HERO = (
    '<div class="landing-hero">'
    '<div class="landing-hero__brand" role="heading" aria-level="1">'
    "O que você deseja consultar?"
    "</div>"
    '<div class="landing-hero__subtitle">'
    "Você pode consultar limite, pedir aumento, análise de crédito e acompanhar "
    "cotações de moedas."
    "</div>"
    "</div>"
)

_TYPING_INDICATOR = """
<div class="chat-row chat-row--assistant" role="status" aria-label="Digitando">
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
    if _SUGGESTIONS_KEY not in session:
        session[_SUGGESTIONS_KEY] = ()


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
    session[_SUGGESTIONS_KEY] = ()
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


def suggestions_for(
    state: ConversationState,
    responding_agent: Agent | None,
) -> tuple[QuickAction, ...]:
    """Escolhe as perguntas rapidas do especialista que acabou de responder.

    So devolve atalhos quando o servico terminou: a conversa segue viva, o
    cliente esta autenticado, nenhum especialista guarda o turno (`active_agent`
    voltou a triagem) e nao ha oferta pendente esperando sim ou nao. Enquanto o
    passo atual espera um dado - CPF, valor do limite, resposta da entrevista ou
    moeda - sugerir outro assunto atrapalharia a coleta.
    """
    if state.ended or not state.authenticated:
        return ()
    if state.active_agent is not Agent.TRIAGE or state.pending_flow is not None:
        return ()
    agent = responding_agent if responding_agent is not None else Agent.TRIAGE
    return _AGENT_SUGGESTIONS.get(agent, _AGENT_SUGGESTIONS[Agent.TRIAGE])


def remember_suggestions(
    session: MutableMapping[str, object],
    suggestions: tuple[QuickAction, ...],
) -> None:
    """Guarda so as chaves das perguntas rapidas oferecidas no turno.

    O Streamlit reexecuta o script a cada rerun e redefine `QuickAction`, entao
    um objeto guardado na sessao pertence a classe da execucao anterior e nao
    sobrevive a um `isinstance`. A chave e texto e atravessa o rerun intacta.
    """
    session[_SUGGESTIONS_KEY] = tuple(item.key for item in suggestions)


def stored_suggestions(
    session: MutableMapping[str, object],
) -> tuple[QuickAction, ...]:
    """Devolve as perguntas rapidas do ultimo turno, vazias por padrao."""
    stored = session.get(_SUGGESTIONS_KEY)
    if not isinstance(stored, tuple):
        return ()
    return tuple(
        _SUGGESTION_BY_KEY[key]
        for key in stored
        if isinstance(key, str) and key in _SUGGESTION_BY_KEY
    )


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


def return_to_landing(session: MutableMapping[str, object]) -> None:
    """Volta para a tela inicial sem descartar o atendimento em andamento.

    Nada da conversa e perdido: um atalho ou uma frase na tela inicial retoma o
    mesmo atendimento, ja autenticado, no ponto em que ele parou.
    """
    session[_VIEW_KEY] = LANDING_VIEW
    session.pop(_PENDING_KEY, None)
    remember_suggestions(session, ())


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


def chat_bubble_html(role: str, text: str) -> str:
    """Monta a linha da fala, escapada para impedir HTML vindo da conversa.

    O cliente fala dentro de um balão alinhado à direita; o especialista
    escreve direto na página, à esquerda. Sem avatar: o alinhamento já diz
    quem fala, e o papel continua legível na classe da linha.
    """
    bubble_role = "user" if role == "user" else "assistant"
    body_class = "chat-bubble" if bubble_role == "user" else "chat-text"
    safe_text = escape(text).replace("\n", "<br>")
    return (
        f'<div class="chat-row chat-row--{bubble_role}">'
        f'<div class="{body_class} {body_class}--{bubble_role}">'
        f"{safe_text}</div></div>"
    )


def typing_hold_seconds(elapsed_seconds: float) -> float:
    """Diz quanto falta para os pontos completarem o tempo minimo visivel.

    Turno lento nao ganha espera nenhuma: o piso so cobre a diferenca.
    """
    return max(0.0, _TYPING_MIN_SECONDS - elapsed_seconds)


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
        exchange=ExchangeService(
            AwesomeApiClient(
                settings.awesomeapi_base_url,
                token=_optional_secret(settings.awesomeapi_token),
            )
        ),
        knowledge=KnowledgeService(
            score_limits=ScoreLimitCsvRepository(settings.data_dir / "score_limite.csv")
        ),
        llm=llm,
    )
    return ConversationService(build_graph(dependencies))


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
    remember_suggestions(session, ())
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
            "Use o botão abaixo para iniciar um novo atendimento."
        )
        session[_NOTICE_KEY] = notice
        return notice
    except Exception:
        notice = "Não foi possível processar sua mensagem agora. Tente novamente."
        session[_NOTICE_KEY] = notice
        return notice
    session[_CONVERSATION_KEY] = turn.state
    session[_NOTICE_KEY] = None
    remember_suggestions(session, suggestions_for(turn.state, turn.responding_agent))
    if first_turn and greets_instead_of_replying(turn.state):
        greeting = stored_welcome(session, welcome_message)
        session[_HISTORY_KEY] = _history_with_reply(turn.history, greeting)
        return greeting
    session[_HISTORY_KEY] = list(turn.history)
    return turn.reply


def _optional_secret(secret: SecretStr | None) -> str | None:
    """Devolve o segredo em texto, ou None quando ele nao foi configurado."""
    if secret is None:
        return None
    value = secret.get_secret_value().strip()
    return value or None


def _optional_llm(settings: Settings) -> GroqStructuredLlm | None:
    """Cria o LLM somente quando há chave configurada, sem exigir rede."""
    api_key = settings.groq_api_key
    if api_key is None or not api_key.get_secret_value().strip():
        return None
    try:
        return GroqStructuredLlm(settings)
    except ValueError:
        return None


def load_cloud_secrets() -> None:
    """Promove segredos do Streamlit Cloud a variaveis de ambiente.

    No Cloud a configuracao chega por `secrets.toml`, e o Streamlit so copia
    esses valores para o ambiente quando alguem os le. Como `Settings` le do
    ambiente, sem esta chamada a chave do provedor existiria no painel e mesmo
    assim o atendimento subiria em modo deterministico, sem aviso. Sem arquivo
    de segredos, como no desenvolvimento local com `.env`, nao faz nada.
    """
    st.secrets.load_if_toml_exists()


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


def _chat_placeholder(session: MutableMapping[str, object]) -> str:
    """Convida a escrever sem competir com as sugestoes exibidas no painel."""
    if stored_suggestions(session):
        return "Ou pergunte outra coisa..."
    return "Digite sua mensagem"


def _render_chat_suggestions(
    session: MutableMapping[str, object],
) -> str | None:
    """Desenha o painel de perguntas rapidas e devolve a escolhida.

    O painel mora na barra inferior, logo acima do campo de texto, para que o
    cliente escolha uma pergunta pronta ou escreva outra coisa sem sair do
    lugar. O "x" dispensa a oferta e deixa so o campo.
    """
    suggestions = stored_suggestions(session)
    if not suggestions:
        return None
    selected: str | None = None
    with st.container(key="chat_suggestions"):
        with st.container(key="chat_suggestions_header"):
            title, dismiss = st.columns((11, 1), vertical_alignment="center")
            title.markdown(
                '<div class="suggestions-title">'
                "Sobre o que você quer conversar agora?</div>",
                unsafe_allow_html=True,
            )
            with dismiss, st.container(key="chat_suggestions_dismiss"):
                dismissed = st.button(
                    "✕",
                    key="chat_suggestions_dismiss_button",
                    help="Dispensar as sugestões",
                    use_container_width=True,
                )
        with st.container(key="chat_suggestion_rows"):
            for suggestion in suggestions[:_SUGGESTION_ROWS]:
                clicked = st.button(
                    suggestion.label,
                    key=f"chat_suggestion_{suggestion.key}",
                    icon=suggestion.icon,
                    use_container_width=True,
                )
                if clicked:
                    selected = suggestion.prompt
    if dismissed:
        remember_suggestions(session, ())
        st.rerun()
    return selected


def _render_landing(
    session: MutableMapping[str, object],
    slot: DeltaGenerator,
) -> None:
    """Mostra a apresentação inicial e abre o chat na primeira interação."""
    with slot.container(key="landing"):
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
    started_at = time.perf_counter()
    try:
        submit_user_message(session, service, user_text, stored_welcome(session))
    finally:
        time.sleep(typing_hold_seconds(time.perf_counter() - started_at))
        typing_placeholder.empty()


def _render_chat(
    session: MutableMapping[str, object],
    service: ConversationServiceLike,
    slot: DeltaGenerator,
) -> None:
    """Mostra o histórico, processa a entrada e oferece novo atendimento."""
    with slot.container():
        _render_chat_body(session, service)


def _render_chat_body(
    session: MutableMapping[str, object],
    service: ConversationServiceLike,
) -> None:
    """Desenha barra inferior, atalho de volta e a conversa em si.

    O atalho de volta fica fora do bloco animado da conversa: uma animacao de
    `transform` no ancestral criaria bloco de contencao e tiraria o botao da
    viewport, quebrando o `position: fixed`.
    """
    with st.container(key="back_to_landing"):
        if st.button(
            "Início",
            key="back_to_landing_button",
            icon="⬅️",
            help="Voltar para a tela inicial",
        ):
            return_to_landing(session)
            st.rerun()
    with st.bottom:
        # As sugestoes vem antes do campo para ficarem acima dele na barra.
        chosen = _render_chat_suggestions(session)
        user_input = st.chat_input(_chat_placeholder(session), key="chat_input")
    if chosen is not None:
        # O clique so agenda a mensagem: o rerun redesenha a conversa sem os
        # atalhos ja usados antes de processar o turno.
        session[_PENDING_KEY] = chosen
        remember_suggestions(session, ())
        st.rerun()
    with st.container(key="chat_view"):
        # Sem cabeçalho nem controles aqui: a marca vive na tela inicial e o
        # chat abre direto na conversa.
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
            # Sem aviso repetindo o fim: o especialista ja se despediu na
            # conversa, e aqui basta o caminho para recomecar.
            with st.container(key="new_service"):
                if st.button(
                    "Iniciar novo atendimento",
                    key="new_service_button",
                    icon="🔄",
                ):
                    reset_conversation(session, stored_welcome(session))
                    st.rerun()


def main() -> None:
    """Escolhe entre a tela inicial e o chat conforme o estado da sessão."""
    st.set_page_config(page_title="Banco Ágil - Atendimento", page_icon="🏦")
    st.markdown(_UI_STYLES, unsafe_allow_html=True)
    session = cast(MutableMapping[str, object], st.session_state)
    # Antes do runtime: e ele quem constroi o Settings a partir do ambiente.
    load_cloud_secrets()
    service, llm = _get_runtime()
    if _HISTORY_KEY not in session:
        init_session(session, generate_welcome_message(llm))
    else:
        init_session(session)

    # Um slot por tela, sempre criados na mesma ordem. O slot da tela inativa
    # fica vazio e apaga, ja no inicio deste run, o que ela desenhou no run
    # anterior. Sem isso o navegador segura aquela arvore ate o run atual
    # terminar - e o run do chat espera os pontos de digitacao, tempo de sobra
    # para o campo e os atalhos da tela inicial aparecerem sobre a conversa.
    # Limpar antes do rerun nao resolve: a fila de deltas pendentes e descartada
    # quando o run seguinte comeca.
    landing_slot = st.empty()
    chat_slot = st.empty()

    if current_view(session) == LANDING_VIEW:
        _render_landing(session, landing_slot)
        return
    _render_chat(session, service, chat_slot)


if __name__ == "__main__" and st.runtime.exists():
    main()
