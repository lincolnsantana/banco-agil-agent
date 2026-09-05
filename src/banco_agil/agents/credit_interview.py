"""No deterministico da entrevista de credito."""

import re

from banco_agil.agents._shared import (
    authentication_reply_if_missing,
    end_reply_if_requested,
    normalize_short_answer,
    parse_confirmation,
)
from banco_agil.agents.state import ConversationState, CreditInterviewDraft
from banco_agil.domain.enums import Agent, Intent
from banco_agil.domain.exceptions import DomainError, RepositoryError
from banco_agil.domain.models import ScoreUpdateResult
from banco_agil.services.credit_interview import (
    CreditInterviewService,
    InterviewField,
)
from banco_agil.tools.banking import update_credit_score

_INTERVIEW_OPENING = (
    "Posso cuidar disso agora. A entrevista de crédito tem cinco perguntas "
    "sobre renda, emprego, despesas, dependentes e dívidas. Com elas recalculo "
    "seu score, sem garantir aprovação, e você pode parar quando quiser, sem "
    "guardar nada. Qual é sua renda mensal?"
)

_INTERVIEW_ABANDONED = (
    "Tudo bem, encerrei a entrevista e não guardei nada. Posso ajudar com "
    "outro serviço ou encerrar o atendimento?"
)

_QUESTIONS = {
    InterviewField.MONTHLY_INCOME: "Qual é sua renda mensal? Informe apenas o valor.",
    InterviewField.EMPLOYMENT_TYPE: (
        "Qual é seu tipo de emprego: formal, autônomo ou desempregado?"
    ),
    InterviewField.MONTHLY_EXPENSES: (
        "Qual é o total das suas despesas fixas mensais?"
    ),
    InterviewField.DEPENDENTS: "Quantos dependentes você possui?",
    InterviewField.ACTIVE_DEBTS: "Você possui dívidas ativas? Responda sim ou não.",
}


def handle_credit_interview(
    state: ConversationState,
    user_text: str,
    service: CreditInterviewService,
) -> str:
    """Coleta consentimento e uma resposta validada por turno."""
    end_reply = end_reply_if_requested(state, user_text)
    if end_reply is not None:
        return end_reply
    authentication_reply = authentication_reply_if_missing(state)
    if authentication_reply is not None:
        return authentication_reply

    if state.interview_draft.consent_given and _cancels_interview(user_text):
        return _cancel_interview(state)

    if not state.interview_draft.consent_given:
        consent = _explicit_interview_consent(user_text)
        if consent is None and _cancels_interview(user_text):
            # Antes do consentimento a recusa passa pelo servico, que e quem
            # detem o consentimento. Sem isto, com a intencao ja reconhecida
            # pela triagem, um "cancelar" aqui iniciaria a coleta.
            consent = False
        if consent is None and not _looks_like_interview_request(user_text):
            consent = parse_confirmation(user_text)
        if consent is None and state.intent is Intent.CREDIT_INTERVIEW:
            # A triagem ja reconheceu o pedido; nos turnos retomados apos a
            # autenticacao o texto aqui e a data de nascimento, nao o pedido.
            consent = True
        if consent is None:
            return "Quer realizar a entrevista de crédito agora?"
        try:
            progress = service.start(state, consent)
        except DomainError:
            state.active_agent = Agent.CREDIT
            return "Não há solicitação de limite pendente para esta entrevista."
        if progress.consent_declined:
            state.requested_limit = None
            state.intent = Intent.UNKNOWN
            state.active_agent = Agent.TRIAGE
            return "Tudo bem. Posso ajudar com outro serviço ou encerrar o atendimento."
        return _INTERVIEW_OPENING

    current_field = _current_field(state)
    try:
        progress = service.collect_answer(state, user_text)
    except DomainError:
        return f"Resposta em formato inválido. {_question(current_field)}"

    if progress.completed_interview is not None:
        try:
            score_result = ScoreUpdateResult.model_validate(
                update_credit_score.invoke(
                    {
                        "interview": progress.completed_interview,
                        "state": state,
                        "service": service,
                    }
                )
            )
        except RepositoryError:
            return "Não foi possível atualizar o score agora. Tente novamente."
        if state.credit_reanalysis_pending:
            return (
                "Entrevista concluída e score atualizado. Farei a reanálise sem "
                "garantia de aprovação."
            )
        return _score_completion_reply(
            score_result.previous_score, score_result.new_score
        )
    return _question(progress.next_field)


def _score_completion_reply(previous_score: int, new_score: int) -> str:
    """Monta a conclusão contextual sem prometer aprovação."""
    if new_score > previous_score:
        return (
            f"Que boa notícia: seu score subiu de {previous_score} para "
            f"{new_score} com os dados atualizados. Deseja continuar ou "
            "encerrar o atendimento?"
        )
    if new_score < previous_score:
        return (
            f"Seu score foi atualizado de {previous_score} para {new_score}, "
            "uma queda pelos dados informados, e isso não define seus próximos "
            "passos. Deseja continuar ou encerrar o atendimento?"
        )
    return (
        f"Seu score permanece em {new_score} após a entrevista. Deseja "
        "continuar ou encerrar o atendimento?"
    )


def _current_field(state: ConversationState) -> InterviewField:
    draft = state.interview_draft
    if draft.monthly_income is None:
        return InterviewField.MONTHLY_INCOME
    if draft.employment_type is None:
        return InterviewField.EMPLOYMENT_TYPE
    if draft.monthly_expenses is None:
        return InterviewField.MONTHLY_EXPENSES
    if draft.dependents is None:
        return InterviewField.DEPENDENTS
    return InterviewField.ACTIVE_DEBTS


_INTERVIEW_DECLINE_MARKERS = frozenset({"nao", "nunca", "jamais", "recuso", "dispenso"})
_INTERVIEW_ACTION_MARKERS = frozenset(
    {"fazer", "realizar", "comecar", "iniciar", "participar", "aceito", "concordo"}
)
_INTERVIEW_INFORMATIONAL_MARKERS = frozenset(
    {"saber", "entender", "conhecer", "informacao", "duvida", "explicar", "funciona"}
)


def _explicit_interview_consent(user_text: str) -> bool | None:
    """Trata pedido explícito de entrevista como consentimento ou recusa.

    O texto precisa mencionar a entrevista; negação explícita nunca vira
    consentimento e pergunta genérica continua pedindo confirmação.
    """
    normalized = normalize_short_answer(user_text)
    if "entrevista" not in normalized:
        return None
    words = set(normalized.split())
    if words & _INTERVIEW_DECLINE_MARKERS:
        return False
    if words & _INTERVIEW_ACTION_MARKERS:
        return True
    if "quero" in words and not (words & _INTERVIEW_INFORMATIONAL_MARKERS):
        return True
    return None


# Palavras que pedem interrupcao. Nenhuma aparece numa resposta valida da
# entrevista, que aceita numero, tipo de emprego ou sim/nao.
_CANCEL_MARKERS = frozenset(
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
_CANCEL_PHRASES = (
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
_CANCEL_ALONE = frozenset({"para", "para."})
# Respostas validas de dividas ativas, que nunca podem ser lidas como recusa.
_DEBT_ANSWERS = frozenset({"sim", "nao"})


def _cancels_interview(user_text: str) -> bool:
    """Detecta pedido de cancelamento em qualquer ponto da entrevista.

    Um "nao" isolado responde a pergunta de dividas ativas e jamais cancela.
    Fora isso a deteccao pode ser generosa: os campos so aceitam numero, tipo
    de emprego ou sim/nao, entao nenhuma palavra daqui colide com resposta
    valida.
    """
    normalized = normalize_short_answer(user_text)
    if normalized in _DEBT_ANSWERS:
        return False
    if normalized in _CANCEL_ALONE:
        return True
    if any(phrase in normalized for phrase in _CANCEL_PHRASES):
        return True
    return bool(frozenset(re.findall(r"[a-z]+", normalized)) & _CANCEL_MARKERS)


def _cancel_interview(state: ConversationState) -> str:
    """Descarta o rascunho e devolve a conversa a triagem.

    Consentimento informado precisa ser revogavel a qualquer momento, entao o
    limite solicitado e a reanalise pendente caem junto com as respostas.
    """
    state.interview_draft = CreditInterviewDraft()
    state.requested_limit = None
    state.credit_reanalysis_pending = False
    state.intent = Intent.UNKNOWN
    state.active_agent = Agent.TRIAGE
    return _INTERVIEW_ABANDONED


def _looks_like_interview_request(user_text: str) -> bool:
    """Evita tratar a intenção original como resposta à pergunta de consentimento."""
    words = set(normalize_short_answer(user_text).split())
    return bool(words & {"aumentar", "melhorar", "rever", "score", "pontuacao"})


def _question(field: InterviewField | None) -> str:
    if field is None:
        raise ValueError("interview progress must provide the next field")
    return _QUESTIONS[field]
