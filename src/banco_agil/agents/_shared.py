"""Comportamentos pequenos compartilhados pelos especialistas."""

import re
import unicodedata
from collections.abc import Sequence
from decimal import Decimal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from banco_agil.agents.state import ConversationState, CreditInterviewDraft
from banco_agil.domain.enums import Agent, EndReason, Intent
from banco_agil.domain.exceptions import IntegrationError
from banco_agil.domain.models import EndServiceResult
from banco_agil.integrations.llm import StructuredLlm
from banco_agil.prompts.renderer import render_prompt
from banco_agil.tools.banking import end_service

_END_REQUESTS = {
    "cancelar atendimento",
    "encerrar",
    "encerrar atendimento",
    "fim",
    "parar",
    "quero encerrar",
    "nao quero continuar",
    "sair",
}
_END_PATTERN = re.compile(
    r"^(?:por favor,?\s*)?"
    r"(?:(?:eu\s+)?(?:quero|desejo|gostaria\s+de|preciso|prefiro)\s+|"
    r"(?:pode|podemos|vamos)\s+)?"
    r"(?:encerrar|encerre|encerra|finalizar|finalize|finaliza|terminar|termine|"
    r"termina|fechar|feche|fecha|parar|sair)"
    r"(?:\s+(?:o|a|este|esta|esse|essa|minha|meu|do|da))?"
    r"(?:\s+(?:atendimento|conversa|chat|sessao|servico))?"
    r"(?:\s+(?:agora|ai|ja|por\s+favor|por\s+aqui))?$"
)
_END_CONTINUATION_PATTERN = re.compile(
    r"^(?:eu\s+)?nao\s+(?:quero|desejo|gostaria\s+de|pretendo)\s+"
    r"(?:mais\s+)?continuar(?:\s+com)?(?:\s+(?:o|a|este|esta|minha|meu))?"
    r"(?:\s+(?:atendimento|conversa|chat|sessao|servico))?$"
)
# Despedida vale como pedido de fim: quem diz tchau nao espera um menu de
# servicos. Cada forma e uma frase inteira, nunca um pedaco de outra.
_FAREWELL_PATTERN = re.compile(
    r"^(?:tchau(?:\s+tchau)?|xau|adeus|falou|"
    r"ate\s+(?:logo|mais|breve|a\s+proxima|mais\s+ver))$"
)
# "era so isso" fecha o assunto; "isso" e "isso mesmo" sozinhos ficam de fora
# de proposito, porque sao confirmacao ("isso!") e encerrariam por engano.
_NOTHING_ELSE_PATTERN = re.compile(
    r"^(?:(?:era|e)\s+)?so\s+isso(?:\s+mesmo)?$"
    r"|^era\s+isso(?:\s+mesmo)?$"
    r"|^isso\s+e\s+tudo$"
    r"|^(?:nao\s+(?:quero|preciso|vou\s+precisar)(?:\s+de)?\s+mais\s+nada)$"
    r"|^(?:sem\s+mais(?:\s+nada)?)$"
)
# Cortesia que emoldura o pedido sem mudar o que ele diz. Retirada das bordas
# antes da comparacao, o que evita repetir cada variacao em todos os padroes.
_COURTESY_EDGE_WORDS = frozenset(
    {
        "obrigado",
        "obrigada",
        "valeu",
        "agradeco",
        "grato",
        "grata",
        "entao",
        "ok",
        "beleza",
        "ai",
        "pode",
    }
)
GREETING_REPLY = (
    "Olá! Tudo bem por aqui. Posso consultar seu limite, analisar um aumento, "
    "atualizar seu score pela entrevista ou ver a cotação de moedas. O que você "
    "prefere?"
)
# Frases inteiras: "oi" cumprimenta, mas "oi, qual meu limite" e um pedido e
# nao pode virar saudacao.
_GREETINGS = frozenset(
    {
        "oi",
        "ola",
        "opa",
        "e ai",
        "eai",
        "hey",
        "bom dia",
        "boa tarde",
        "boa noite",
        "tudo bem",
        "tudo bom",
        "oi tudo bem",
        "ola tudo bem",
        "oi tudo bom",
        "bom dia tudo bem",
        "boa tarde tudo bem",
        "boa noite tudo bem",
    }
)


def is_greeting(user_text: str) -> bool:
    """Reconhece cumprimento isolado, sem pedido junto.

    A pontuacao interna some antes da comparacao: "oi, tudo bem?" e o mesmo
    cumprimento que "oi tudo bem". O casamento segue sendo da frase inteira,
    entao "oi, qual e meu limite" continua sendo um pedido.
    """
    normalized = normalized_text(user_text)
    sem_pontuacao = " ".join(re.sub(r"[.,!?;:'\"()\[\]-]", " ", normalized).split())
    return sem_pontuacao in _GREETINGS


HELP_REPLY = (
    "Claro! Posso consultar seu limite de crédito, solicitar um aumento de "
    "limite, conduzir a entrevista de crédito para revisar seu score e "
    "consultar cotações de moedas como dólar e euro. Por onde quer começar?"
)

_HELP_PHRASES = (
    "o que voce pode fazer",
    "o que voce pode realizar",
    "o que voce faz",
    "o que voce oferece",
    "o que voce realiza",
    "o que pode realizar",
    "o que sabe fazer",
    "quais servicos",
    "que servicos",
    "quais opcoes",
    "suas funcionalidades",
    "suas funcoes",
    "como funciona",
    "me fale sobre os servicos",
    "o que posso fazer",
)


def is_help_request(user_text: str) -> bool:
    """Detecta pergunta sobre o atendimento sem usar LLM."""
    normalized = normalized_text(user_text)
    if normalized in {"ajuda", "menu"}:
        return True
    return any(phrase in normalized for phrase in _HELP_PHRASES)


_HOWTO_MARKERS = (
    "como faco",
    "como fazer",
    "como posso",
    "como consigo",
    "como realizo",
    "como solicitar",
    "como solicito",
    "como pedir",
    "como peco",
    "como consultar",
    "como consulto",
    "como aumento",
    "como aumentar",
    "como melhorar",
    "como ver",
    "como vejo",
    "como funciona",
    "como comeco",
    "como iniciar",
    "passo a passo",
    "quais sao os passos",
    "o que preciso para",
    "posso aumentar",
    "e possivel aumentar",
    "tem como aumentar",
    "por que nao consigo aumentar",
    "porque nao consigo aumentar",
)

_FLOW_AFFIRMATIVE_WORDS = frozenset(
    {
        "sim",
        "quero",
        "vamos",
        "claro",
        "gostaria",
        "aceito",
        "concordo",
        "bora",
        "confirmo",
        "confirmado",
        "fechado",
        "prossiga",
        "continue",
        "pode",
        "faca",
        "realize",
        "manda",
        "interesse",
        "topo",
        "ok",
        "okay",
        "beleza",
        "certo",
    }
)
_FLOW_REFUSAL_WORDS = frozenset(
    {
        "quero",
        "vou",
        "prefiro",
        "obrigado",
        "obrigada",
        "valeu",
        "dispenso",
        "recuso",
        "cancela",
        "cancelar",
        "deixa",
        "interesse",
        "pode",
        "precisa",
    }
)
_FLOW_NEVER_WORDS = frozenset({"nunca", "jamais"})


_WORD_PATTERN = re.compile(r"[a-z0-9]+")
_CURRENCY_WORDS = frozenset(
    {
        "cambio",
        "cotacao",
        "cotacoes",
        "dolar",
        "euro",
        "libra",
        "iene",
        "moeda",
        "moedas",
    }
)
_LIMIT_NOUNS = frozenset({"limite", "limites", "credito", "cartao"})
_SCORE_NOUNS = frozenset({"score", "pontuacao", "pontos"})
# Pedem aumento sozinhos: "quero aumentar" ja e pedido, sem substantivo.
_STRONG_INCREASE_MARKERS = frozenset(
    {
        "aumentar",
        "aumento",
        "aumenta",
        "aumente",
        "aumentando",
        "subir",
        "suba",
        "elevar",
        "eleve",
        "ampliar",
        "amplie",
    }
)
# Só qualificam algo: sem substantivo, "mais" ou "novo" nao dizem nada.
_QUALIFIER_MARKERS = frozenset(
    {
        "melhorar",
        "melhore",
        "melhorou",
        "maior",
        "mais",
        "novo",
        "nova",
        "alterar",
        "altere",
        "ajustar",
        "ajuste",
        "modificar",
        "modifique",
        "mudar",
        "mude",
        "atualizar",
        "atualize",
        "revisar",
        "rever",
        "liberar",
        "libere",
    }
)
_INCREASE_MARKERS = _STRONG_INCREASE_MARKERS | _QUALIFIER_MARKERS
_QUERY_MARKERS = frozenset(
    {
        "qual",
        "quais",
        "quanto",
        "quantos",
        "ver",
        "vejo",
        "veja",
        "consultar",
        "consulto",
        "consulte",
        "consulta",
        "mostrar",
        "mostra",
        "mostre",
        "visualizar",
        "visualizo",
        "saber",
        "conferir",
        "confiro",
        "checar",
        "exibir",
        "exiba",
        "informar",
    }
)


def _first_index(words: list[str], vocabulary: frozenset[str]) -> int | None:
    for index, word in enumerate(words):
        if word in vocabulary:
            return index
    return None


def _increase_targets_score(
    increase_at: int,
    limit_at: int | None,
    score_at: int | None,
) -> bool:
    """Decide se o pedido de aumento recai sobre o score ou sobre o limite."""
    if score_at is None:
        return False
    if limit_at is None:
        return True
    return abs(score_at - increase_at) < abs(limit_at - increase_at)


def classify_banking_request(user_text: str) -> Intent | None:
    """Classifica o pedido pelo alvo da acao, nao por palavra solta.

    Uma varredura plana confunde "aumentar meu limite porque o score melhorou"
    com pedido de score, e "saber meu limite antes de pedir aumento" com pedido
    de aumento. Aqui a acao do cliente, consultar ou aumentar, e casada com o
    substantivo mais proximo dela; sem acao reconhecida, o pedido e consulta.
    """
    words = _WORD_PATTERN.findall(normalized_text(user_text))
    if not words:
        return None
    if _first_index(words, _CURRENCY_WORDS) is not None:
        return Intent.EXCHANGE_RATE
    if "entrevista" in words:
        return Intent.CREDIT_INTERVIEW

    limit_at = _first_index(words, _LIMIT_NOUNS)
    score_at = _first_index(words, _SCORE_NOUNS)
    if limit_at is None and score_at is None:
        # "quero aumentar", sem dizer o que, e pedido de limite neste banco.
        if _first_index(words, _STRONG_INCREASE_MARKERS) is not None:
            return Intent.LIMIT_INCREASE
        return None

    consultation = (
        Intent.CREDIT_LIMIT if limit_at is not None else Intent.CREDIT_INTERVIEW
    )
    increase_at = _first_index(words, _INCREASE_MARKERS)
    if increase_at is None:
        return consultation

    query_at = _first_index(words, _QUERY_MARKERS)
    if query_at is not None and query_at < increase_at:
        # "quero saber meu limite antes de pedir aumento" continua consulta.
        return consultation
    if _increase_targets_score(increase_at, limit_at, score_at):
        return Intent.CREDIT_INTERVIEW
    return Intent.LIMIT_INCREASE


def detect_howto_topic(user_text: str) -> Intent | None:
    """Mapeia pergunta de como-fazer ao fluxo, sem usar LLM."""
    normalized = normalized_text(user_text)
    if not any(marker in normalized for marker in _HOWTO_MARKERS):
        return None
    return classify_banking_request(user_text)


# Abrem duvida sobre o assunto, em vez de pedir uma operacao.
_INFORMATION_MARKERS = (
    "por que",
    "porque",
    "por quais",
    "o que e",
    "o que sao",
    "o que acontece",
    "o que significa",
    "que significa",
    "de onde",
    "quanto tempo",
    "quantas perguntas",
    "voces cobram",
    "voces fazem",
    "voces tem",
    "voces guardam",
    "existe taxa",
    "tem taxa",
    "tem custo",
    "e cobrado",
    "e seguro",
    "e bom",
    "e ruim",
    "vale a pena",
    "posso pedir",
    "posso fazer",
    "preciso informar",
    "precisa informar",
    "como funciona",
    "como voces",
    "como e calculado",
    "como e feito",
    # Pedem explicacao, e nao a operacao: "me explica o score" quer entender,
    # nao iniciar a entrevista. Ficam aqui, e nao entre os marcadores de
    # como-fazer, porque aqueles levam direto ao fluxo.
    "me explica",
    "me explique",
    "quem e voce",
    "qual seu nome",
    "qual e o seu nome",
    "como voce se chama",
    "voce e um robo",
    "voce e humano",
    "com quem estou falando",
)
# Pedem execucao: prevalecem mesmo com verniz de pergunta. "pedir" fica de
# fora de proposito, porque aparece tanto em pergunta quanto em pedido, como
# em "posso pedir aumento de novo?".
_COMMAND_MARKERS = frozenset(
    {
        "quero",
        "queria",
        "gostaria",
        "solicitar",
        "solicito",
        "faca",
        "faz",
        "abre",
        "abrir",
        "inicia",
        "iniciar",
        "comeca",
        "comecar",
    }
)


def detects_information_question(user_text: str) -> bool:
    """Indica que o cliente quer entender algo, nao executar uma operacao.

    Sem isto, todo substantivo bancario vira acao e o atendimento responde
    "qual limite voce quer?" para quem perguntou por que um pedido foi negado.
    Marcador de comando vence: "quero pedir aumento" continua sendo pedido,
    mesmo contendo "posso pedir".
    """
    normalized = normalized_text(user_text)
    words = frozenset(_WORD_PATTERN.findall(normalized))
    if words & _COMMAND_MARKERS:
        return False
    return any(marker in normalized for marker in _INFORMATION_MARKERS)


_FLOW_AFFIRMATIVE_PHRASES = frozenset(
    {
        "sim",
        "quero",
        "vamos",
        "claro",
        "com certeza",
        "pode ser",
        "gostaria",
        "aceito",
        "concordo",
        "bora",
        "confirmo",
        "confirmado",
        "fechado",
        "prossiga",
        "continue",
        "pode",
        "pode prosseguir",
        "pode continuar",
        "pode fazer",
        "faca isso",
        "vamos fazer",
        "vamos nessa",
        "tenho interesse",
        "quero realizar",
        "quero fazer",
        "manda ver",
        "manda bala",
        "eu topo",
        "ok",
        "okay",
        "beleza",
        "certo",
        "isso",
        "isso mesmo",
    }
)
_FLOW_NEGATIVE_PHRASES = frozenset(
    {
        "nao",
        "agora nao",
        "depois",
        "prefiro nao",
        "dispenso",
        "cancela",
        "cancelar",
        "melhor nao",
        "deixa",
        "deixa pra la",
        "deixa para depois",
        "mais tarde",
        "outro momento",
        "nao agora",
        "nao tenho interesse",
        "nao precisa",
    }
)


def parse_flow_answer(user_text: str) -> bool | None:
    """Converte resposta ampla à confirmação de fluxo, sem usar LLM."""
    normalized = normalize_short_answer(user_text)
    if normalized in _FLOW_NEGATIVE_PHRASES:
        return False
    if normalized in _FLOW_AFFIRMATIVE_PHRASES:
        return True
    words = set(normalized.split())
    if "nao" in words and words & _FLOW_REFUSAL_WORDS:
        return False
    if words & _FLOW_AFFIRMATIVE_WORDS:
        return True
    if words & _FLOW_NEVER_WORDS:
        return False
    return None


HANDOFF_REPLY = "Certo. Vou prosseguir com sua solicitação."
REFUSAL_REPLY = (
    "Tudo bem, deixo isso de lado. Posso ajudar com limite de crédito, "
    "entrevista de crédito ou cotação de moedas. O que deseja?"
)

# Pedidos que um especialista pode assumir no meio do fluxo de outro.
RESUMABLE_INTENTS = frozenset(
    {
        Intent.CREDIT_LIMIT,
        Intent.LIMIT_INCREASE,
        Intent.CREDIT_INTERVIEW,
        Intent.EXCHANGE_RATE,
    }
)
# Palavras que pedem interrupcao do passo atual. Nenhuma aparece numa resposta
# valida de valor, moeda, tipo de emprego ou sim/nao.
_REFUSAL_MARKERS = frozenset(
    {
        "cancelar",
        "cancela",
        "cancele",
        "cancelo",
        "cancelamento",
        "parar",
        "pare",
        "chega",
        "basta",
        "esquece",
        "esqueca",
        "esquecer",
        "desisto",
        "desisti",
        "desistir",
        "interromper",
        "interrompe",
    }
)
_REFUSAL_PHRASES = (
    "nao quero mais",
    "nao quero fazer",
    "nao quero continuar",
    "nao quero responder",
    "nao quero seguir",
    "nao quero isso",
    "prefiro nao",
    "melhor nao",
    "agora nao",
    "deixa pra la",
    "deixa para la",
    "deixa quieto",
    "outra hora",
    "mais tarde",
    "depois eu faco",
)
# "para" sozinho pede parada; dentro de uma frase e preposicao comum.
_REFUSAL_ALONE = frozenset({"para"})


class FlowChange(BaseModel):
    """Recusa do passo atual ou pedido novo, sem escolher agente.

    Vem do parser ou do entendimento do LLM ja aterrado; em ambos os casos o
    especialista de destino revalida antes de agir.
    """

    declines_current: bool = False
    requested_intent: Intent = Intent.UNKNOWN


def agent_for_intent(intent: Intent) -> Agent:
    """Traduz a intencao de atendimento no especialista responsavel."""
    if intent is Intent.EXCHANGE_RATE:
        return Agent.EXCHANGE
    if intent is Intent.CREDIT_INTERVIEW:
        return Agent.CREDIT_INTERVIEW
    return Agent.CREDIT


def detects_refusal(user_text: str) -> bool:
    """Detecta recusa ou desistencia do passo atual sem usar LLM.

    Generosa de proposito: onde e chamada, o texto ja deixou de ser resposta
    valida ao passo. Um "nao" isolado conta como recusa aqui; a entrevista, que
    aceita "nao" como resposta de dividas, trata essa excecao antes de chamar.
    """
    normalized = normalize_short_answer(user_text)
    if normalized in _REFUSAL_ALONE:
        return True
    if any(phrase in normalized for phrase in _REFUSAL_PHRASES):
        return True
    if frozenset(_WORD_PATTERN.findall(normalized)) & _REFUSAL_MARKERS:
        return True
    return parse_flow_answer(user_text) is False


def parse_flow_change(user_text: str, current_intent: Intent) -> FlowChange | None:
    """Reconhece por parser um pedido diferente do fluxo atual ou uma recusa."""
    requested = classify_banking_request(user_text)
    if requested is not None and requested is not current_intent:
        return FlowChange(requested_intent=requested)
    if requested is None and detects_refusal(user_text):
        return FlowChange(declines_current=True)
    return None


def reset_flow(state: ConversationState) -> None:
    """Descarta o passo em andamento: rascunho, limite pedido e oferta pendente.

    Recusar no meio da coleta retira o consentimento, entao respostas parciais
    da entrevista nao sobrevivem, e a reanalise que dependia delas tambem nao.
    """
    state.interview_draft = CreditInterviewDraft()
    state.requested_limit = None
    state.credit_reanalysis_pending = False
    state.pending_flow = None


def apply_flow_change(state: ConversationState, change: FlowChange) -> str:
    """Aplica recusa ou redirecionamento e devolve a resposta canonica.

    Pedido novo entrega o turno ao especialista dele, que substitui esta
    resposta no mesmo turno; recusa sem pedido devolve a conversa a triagem.
    """
    reset_flow(state)
    if change.requested_intent is Intent.HELP:
        state.intent = Intent.UNKNOWN
        state.active_agent = Agent.TRIAGE
        return HELP_REPLY
    if change.requested_intent in RESUMABLE_INTENTS:
        state.intent = change.requested_intent
        state.active_agent = agent_for_intent(change.requested_intent)
        return HANDOFF_REPLY
    state.intent = Intent.UNKNOWN
    state.active_agent = Agent.TRIAGE
    return REFUSAL_REPLY


_FACT_PATTERN = re.compile(
    r"\b[A-Z]{3}-[A-Z]{3}\b|\b\d{4}-\d{2}-\d{2}\b|(?:R\$\s*)?(?:\d[\d.,]*\d|\d)"
)
_FACT_TOKEN_PATTERN = re.compile(r"\[DADO_\d+\]")
# Travessao e meia-risca; o hifen comum fica de fora de proposito.
_DASH_PATTERN = re.compile(r"\s*[\u2013\u2014]\s*")
_USER_TEXT_LIMIT = 280
_MASKED_NUMBER = "esse valor"
_CONTROL_PATTERN = re.compile(r"[\x00-\x1f\x7f]+")
# Colchetes sairiam caros: o cliente poderia forjar um marcador [DADO_N].
_STRUCTURE_PATTERN = re.compile(r"[`<>{}\[\]|]+")
_CPF_TEXT_PATTERN = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
_DATE_TEXT_PATTERN = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b")
_NUMBER_TEXT_PATTERN = re.compile(r"(?:R\$\s*)?\d[\d.,]*")
_LEAK_PHRASES = (
    "system prompt",
    "prompt do sistema",
    "instrucao de sistema",
    "como modelo de linguagem",
    "minhas instrucoes",
    "regras internas",
    "contexto seguro",
    "pergunta do cliente",
    "texto validado",
    "dado_n",
)
# Teto da reescrita. Precisa caber o canonico mais o reconhecimento do que o
# cliente disse; apertado demais, o modelo devolve o canonico sem mudanca.
_MAX_REWRITE_LENGTH = 480


class RewrittenReply(BaseModel):
    """Reescrita integral da resposta canônica, com fatos mascarados."""

    reply: str = Field(min_length=1, max_length=_MAX_REWRITE_LENGTH)


def normalized_text(value: str) -> str:
    """Normaliza texto curto para parsers deterministicos."""
    without_accents = "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold().strip())
        if not unicodedata.combining(character)
    )
    return " ".join(without_accents.split())


def _without_courtesy(normalized: str) -> str:
    """Remove cortesia das bordas para o pedido ser comparado pelo que diz.

    "pode encerrar, obrigado" e "encerrar" sao o mesmo pedido, e sem esta poda
    cada padrao teria de repetir todas as combinacoes. Cortesia sozinha nao
    sobra nada, e string vazia nao casa com nenhum padrao: agradecer no meio do
    atendimento continua nao encerrando.
    """
    palavras = normalized.replace(",", " ").split()
    inicio, fim = 0, len(palavras)
    while inicio < fim and palavras[inicio] in _COURTESY_EDGE_WORDS:
        inicio += 1
    while fim > inicio and palavras[fim - 1] in _COURTESY_EDGE_WORDS:
        fim -= 1
    podado = " ".join(palavras[inicio:fim])
    # "por favor" e par: some das duas bordas.
    podado = re.sub(r"^por favor\s+|\s+por favor$", "", podado).strip()
    return podado


def end_reply_if_requested(state: ConversationState, user_text: str) -> str | None:
    """Encerra o atendimento antes de qualquer outra operacao.

    O pedido de fim vale em qualquer no e a qualquer momento, entao a deteccao
    precisa alcancar a linguagem real: despedida, cortesia em volta do pedido e
    "era so isso". Continua por frase inteira, nunca por pedaco: "quero sair das
    dividas" segue sendo um assunto, nao uma saida.
    """
    normalized = normalized_text(user_text).strip(".,!?;:'\"()[]-").strip()
    candidatos = {normalized, _without_courtesy(normalized)}
    if not any(
        candidato in _END_REQUESTS
        or _END_PATTERN.fullmatch(candidato) is not None
        or _END_CONTINUATION_PATTERN.fullmatch(candidato) is not None
        or _FAREWELL_PATTERN.fullmatch(candidato) is not None
        or _NOTHING_ELSE_PATTERN.fullmatch(candidato) is not None
        for candidato in candidatos
        if candidato
    ):
        return None
    end_conversation(state, EndReason.USER_REQUEST)
    return "Atendimento encerrado. Obrigado pela conversa e volte quando precisar."


def end_conversation(state: ConversationState, reason: EndReason) -> None:
    """Executa a tool de encerramento e aplica seu resultado ao estado.

    Descarta tambem a entrevista em andamento: quem pede para encerrar no meio
    da coleta esta retirando o consentimento, e renda, despesas e dividas
    parciais nao podem sobreviver ao pedido.
    """
    result = EndServiceResult.model_validate(end_service.invoke({"reason": reason}))
    state.interview_draft = CreditInterviewDraft()
    state.requested_limit = None
    state.credit_reanalysis_pending = False
    state.end(result.reason)


def authentication_reply_if_missing(state: ConversationState) -> str | None:
    """Redireciona operacao protegida para coleta de credenciais."""
    if state.authenticated:
        return None
    state.active_agent = Agent.TRIAGE
    return "Para continuar, preciso confirmar seus dados. Por favor, informe seu CPF."


_SHORT_ANSWER_PUNCTUATION = ".,!?;:'\"()[]-"


def normalize_short_answer(value: str) -> str:
    """Remove acentos, caixa e pontuação lateral de respostas curtas."""
    return normalized_text(value).strip(_SHORT_ANSWER_PUNCTUATION).strip()


def parse_confirmation(value: str) -> bool | None:
    """Converte consentimento expresso em linguagem natural sem usar LLM."""
    return parse_flow_answer(value)


def mask_user_text(value: str) -> str:
    """Preserva a pergunta do cliente sem PII, numero ou marcacao estrutural.

    A redacao dos especialistas precisa do texto real para responder no tom de
    quem perguntou. O que nao pode viajar e o dado sensivel: CPF e nascimento
    somem, numeros viram termo neutro e caracteres de controle e de estrutura
    sao descartados para nao abrirem espaco a injecao de instrucao.
    """
    without_control = _CONTROL_PATTERN.sub(" ", value)
    without_pii = _CPF_TEXT_PATTERN.sub(" ", without_control)
    without_pii = _DATE_TEXT_PATTERN.sub(" ", without_pii)
    without_numbers = _NUMBER_TEXT_PATTERN.sub(_MASKED_NUMBER, without_pii)
    collapsed = " ".join(_STRUCTURE_PATTERN.sub(" ", without_numbers).split())
    return collapsed[:_USER_TEXT_LIMIT].strip()


def humanize_reply(
    state: ConversationState,
    canonical_reply: str,
    llm: StructuredLlm | None,
    turn_id: str,
    *,
    responding_agent: Agent | None,
    recent_messages: Sequence[BaseMessage],
    user_text: str,
) -> str:
    """Reescreve a resposta canônica sem permitir mudança nos fatos.

    O LLM recebe o canônico com fatos mascarados e pode redigir a resposta
    final completa. Falhas, saídas inseguras e orçamento consumido retornam
    silenciosamente ao texto determinístico.
    """
    if (
        llm is None
        or not turn_id
        or state.ended
        or responding_agent is None
        or responding_agent is Agent.TRIAGE
        or llm.calls_remaining(turn_id) == 0
    ):
        return canonical_reply

    masked_reply, facts = _mask_facts(canonical_reply)
    prompt_state = state.model_copy(deep=True)
    prompt_state.active_agent = responding_agent
    rendered = render_prompt(prompt_state)
    safe_user_text = mask_user_text(user_text) or "pedido bancario"
    messages: list[BaseMessage] = [
        rendered.system_message,
        *safe_history(recent_messages),
        HumanMessage(
            content=(
                "Você está conversando com o cliente. Escreva a próxima fala "
                "com suas palavras, natural como a de um atendente humano: "
                "retome o que ele disse, no tom dele, e diga o mesmo que o "
                "texto validado diz. Reescreva a forma, nunca o conteúdo. Cada "
                "marcador [DADO_N] aparece igual, e não entram números, fatos, "
                "decisões, promessas ou perguntas que o texto validado não "
                "tenha. Evite repetir as frases dele palavra por palavra. "
                f"O cliente disse: {safe_user_text} "
                f"Texto validado: {masked_reply}"
            )
        ),
    ]
    try:
        result = llm.invoke_structured(
            turn_id,
            messages,
            RewrittenReply,
            prompt_version=rendered.prompt_version,
        )
    except IntegrationError:
        return canonical_reply

    rewritten = normalize_dashes(result.reply)
    restored = _restore_facts(rewritten, facts)
    if restored is None or not _preserves_decision(restored, canonical_reply):
        return canonical_reply
    return restored


def normalize_dashes(text: str) -> str:
    """Troca travessao por virgula sem deixar pontuacao duplicada.

    Instruir o modelo nao basta: o travessao e habito forte de LLM, entao a
    regra de estilo vale deterministicamente sobre a saida ja aceita. Hifen
    comum fica intacto, para nao quebrar pares de moeda como USD-BRL.
    """
    replaced = _DASH_PATTERN.sub(", ", text)
    replaced = re.sub(r"\s+,", ",", replaced)
    replaced = re.sub(r",(?:\s*,)+", ", ", replaced)
    replaced = re.sub(r",\s*([.!?])", r"\1", replaced)
    return " ".join(replaced.split())


def _mask_facts(text: str) -> tuple[str, dict[str, str]]:
    """Substitui pares, datas e números por marcadores opacos."""
    facts: dict[str, str] = {}

    def _replace(match: re.Match[str]) -> str:
        token = f"[DADO_{len(facts) + 1}]"
        facts[token] = match.group()
        return token

    return _FACT_PATTERN.sub(_replace, text), facts


def _restore_facts(masked_reply: str, facts: dict[str, str]) -> str | None:
    """Recoloca os fatos se todos os marcadores forem preservados."""
    if set(_FACT_TOKEN_PATTERN.findall(masked_reply)) != set(facts):
        return None
    restored = masked_reply
    for token, value in facts.items():
        restored = restored.replace(token, value)
    if _FACT_TOKEN_PATTERN.search(restored) is not None:
        return None
    return restored


def _preserves_decision(rewritten: str, canonical_reply: str) -> bool:
    """Exige subconjunto de números, perguntas e ausência de vazamento."""
    normalized_rewritten = normalized_text(rewritten)
    normalized_canonical = normalized_text(canonical_reply)
    if any(leak in normalized_rewritten for leak in _LEAK_PHRASES):
        return False
    if "indisponivel" in normalized_canonical and not any(
        marker in normalized_rewritten
        for marker in ("indisponivel", "nao consegui", "nao foi possivel")
    ):
        return False
    if rewritten.strip().endswith("?") != canonical_reply.strip().endswith("?"):
        return False
    canonical_digits = re.findall(r"\d+", canonical_reply)
    rewritten_digits = re.findall(r"\d+", rewritten)
    remaining = list(canonical_digits)
    for digits in rewritten_digits:
        if digits not in remaining:
            return False
        remaining.remove(digits)
    return True


def safe_history(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Mascara as mensagens recentes antes de enviá-las ao LLM."""
    safe_messages: list[BaseMessage] = []
    for message in messages[-5:]:
        safe_content = mask_user_text(str(message.content))
        if not safe_content:
            continue
        message_type = AIMessage if isinstance(message, AIMessage) else HumanMessage
        safe_messages.append(message_type(content=safe_content))
    return safe_messages


def format_money(value: Decimal) -> str:
    """Formata valor decimal para exibicao bancaria em pt-BR."""
    formatted = f"{value:,.2f}"
    return formatted.replace(",", "_").replace(".", ",").replace("_", ".")
