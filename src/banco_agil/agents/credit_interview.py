"""No deterministico da entrevista de credito."""

import re
from collections.abc import Sequence

from langchain_core.messages import BaseMessage

from banco_agil.agents._shared import (
    apply_flow_change,
    authentication_reply_if_missing,
    detects_refusal,
    end_reply_if_requested,
    normalize_short_answer,
    parse_confirmation,
    parse_flow_change,
)
from banco_agil.agents.knowledge import explanation_for_pending_step
from banco_agil.agents.state import ConversationState, CreditInterviewDraft
from banco_agil.agents.understanding import (
    TurnContext,
    TurnUnderstanding,
    resolve_context,
)
from banco_agil.domain.enums import Agent, Intent
from banco_agil.domain.exceptions import DomainError, RepositoryError
from banco_agil.domain.models import ScoreUpdateResult
from banco_agil.integrations.llm import StructuredLlm
from banco_agil.services.credit_interview import (
    CreditInterviewService,
    InterviewField,
    InterviewProgress,
)
from banco_agil.services.knowledge import KnowledgeService
from banco_agil.tools.banking import update_credit_score

_DIGIT_PATTERN = re.compile(r"\d")

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
    *,
    knowledge: KnowledgeService | None = None,
    context: TurnContext | None = None,
    llm: StructuredLlm | None = None,
    turn_id: str = "",
    recent_messages: Sequence[BaseMessage] = (),
) -> str:
    """Coleta consentimento e respostas validadas, uma ou varias por turno.

    Score e consentimento continuam deterministicos. O LLM entra quando o
    texto nao e resposta valida: pode ler desistencia, pedido de outro
    servico, varias respostas ditas de uma vez ("ganho cinco mil e sou
    formal") ou redigir a pergunta de esclarecimento. Cada valor extraido
    passa pela mesma validacao do servico antes de entrar no rascunho.
    """
    context = resolve_context(context, llm, turn_id, recent_messages)
    end_reply = end_reply_if_requested(state, user_text)
    if end_reply is not None:
        return end_reply
    authentication_reply = authentication_reply_if_missing(state)
    if authentication_reply is not None:
        return authentication_reply

    if not state.interview_draft.consent_given:
        return _handle_consent(state, user_text, service, context)

    # Pedido de outro servico vence o cancelamento generico: "chega, quero ver
    # o dolar" leva o dolar junto em vez de parar na triagem.
    change = parse_flow_change(user_text, Intent.CREDIT_INTERVIEW)
    if change is not None and not change.declines_current:
        return apply_flow_change(state, change)
    if _cancels_interview(user_text):
        return _cancel_interview(state)

    current_field = _current_field(state)
    doubt_reply = _answer_doubt(user_text, knowledge, state, current_field)
    if doubt_reply is not None:
        return doubt_reply
    try:
        progress = service.collect_answer(state, user_text)
    except DomainError:
        return _handle_invalid_answer(state, user_text, service, context, current_field)
    return _advance(state, service, progress)


def _answer_doubt(
    user_text: str,
    knowledge: KnowledgeService | None,
    state: ConversationState,
    current_field: InterviewField,
) -> str | None:
    """Responde duvida sobre a coleta e repete a pergunta do passo.

    Vem antes de qualquer leitura da resposta: sem isto o texto chegava ao LLM
    como resposta invalida e podia ser lido como desistencia, descartando a
    entrevista de quem so queria entender por que o dado e pedido.
    """
    if knowledge is None:
        return None
    return explanation_for_pending_step(
        user_text,
        knowledge,
        state.authenticated_client,
        _question(current_field),
        default_topic="interview_data_use",
    )


def _handle_invalid_answer(
    state: ConversationState,
    user_text: str,
    service: CreditInterviewService,
    context: TurnContext,
    current_field: InterviewField,
) -> str:
    """Le com o LLM uma resposta que o parser do campo rejeitou.

    "sim" e "nao" fora de lugar sao so resposta mal posicionada, nunca recusa:
    um "nao" solto responde dividas ativas e a regra vale em qualquer campo.
    """
    invalid_reply = f"Resposta em formato inválido. {_question(current_field)}"
    if normalize_short_answer(user_text) in _DEBT_ANSWERS:
        return invalid_reply
    understanding = context.understand(user_text, Intent.CREDIT_INTERVIEW)
    if understanding is None:
        return invalid_reply
    change = understanding.flow_change(Intent.CREDIT_INTERVIEW)
    if change is not None and change.declines_current:
        return _cancel_interview(state)
    if change is not None:
        return apply_flow_change(state, change)
    progress = _collect_extracted(state, service, understanding)
    if progress is not None:
        return _advance(state, service, progress)
    return understanding.clarification or invalid_reply


def _handle_consent(
    state: ConversationState,
    user_text: str,
    service: CreditInterviewService,
    context: TurnContext,
) -> str:
    """Le consentimento, recusa ou pedido de outro servico antes de coletar.

    Antes do consentimento a recusa passa pelo servico, que e quem o detem.
    Sem isto, com a intencao ja reconhecida pela triagem, um "cancelar" aqui
    iniciaria a coleta. Pedido de outro servico entrega o turno a ele. Quem ja
    responde a primeira pergunta junto com o "sim" consentiu e e atendido.
    """
    consent = _explicit_interview_consent(user_text)
    if consent is None:
        change = parse_flow_change(user_text, Intent.CREDIT_INTERVIEW)
        if change is not None and not change.declines_current:
            return apply_flow_change(state, change)
        if change is not None:
            consent = False
    if consent is None and not _looks_like_interview_request(user_text):
        consent = parse_confirmation(user_text)
    if consent is None and state.intent is Intent.CREDIT_INTERVIEW:
        # A triagem ja reconheceu o pedido; nos turnos retomados apos a
        # autenticacao o texto aqui e a data de nascimento, nao o pedido.
        consent = True
    understanding = None
    if consent is None:
        understanding = context.understand(user_text, Intent.CREDIT_INTERVIEW)
        if understanding is not None:
            change = understanding.flow_change(Intent.CREDIT_INTERVIEW)
            if change is not None and not change.declines_current:
                return apply_flow_change(state, change)
            if change is not None:
                consent = False
            elif _has_interview_answers(understanding):
                consent = True
    elif consent and _DIGIT_PATTERN.search(user_text):
        # "pode ser, minha renda e 3500": o numero junto do sim e a primeira
        # resposta, e so para aproveita-la o entendimento e consultado.
        understanding = context.understand(user_text, Intent.CREDIT_INTERVIEW)
    if consent is None:
        return (
            understanding.clarification if understanding is not None else None
        ) or "Quer realizar a entrevista de crédito agora?"
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
    if understanding is not None:
        extracted = _collect_extracted(state, service, understanding)
        if extracted is not None:
            return _advance(state, service, extracted)
    return _INTERVIEW_OPENING


def _advance(
    state: ConversationState,
    service: CreditInterviewService,
    progress: InterviewProgress,
) -> str:
    """Conclui a entrevista quando completa; senao faz a proxima pergunta."""
    if progress.completed_interview is None:
        return _question(progress.next_field)
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
    return _score_completion_reply(score_result.previous_score, score_result.new_score)


def _has_interview_answers(understanding: TurnUnderstanding) -> bool:
    return any(
        value is not None
        for value in (
            understanding.monthly_income,
            understanding.employment_type,
            understanding.monthly_expenses,
            understanding.dependents,
            understanding.has_active_debts,
        )
    )


def _collect_extracted(
    state: ConversationState,
    service: CreditInterviewService,
    understanding: TurnUnderstanding,
) -> InterviewProgress | None:
    """Preenche, em ordem, os campos que o cliente respondeu de uma vez.

    Cada valor volta a passar pelo parser do servico como texto canonico, entao
    a validacao e a mesma de uma resposta digitada. Para no primeiro campo sem
    resposta: o especialista pergunta exatamente ele.
    """
    progress: InterviewProgress | None = None
    while True:
        field = _current_field(state)
        answer = _canonical_answer(understanding, field)
        if answer is None:
            return progress
        try:
            progress = service.collect_answer(state, answer)
        except DomainError:
            return progress
        if progress.completed_interview is not None or progress.next_field is None:
            return progress


def _canonical_answer(
    understanding: TurnUnderstanding, field: InterviewField
) -> str | None:
    if field is InterviewField.MONTHLY_INCOME and understanding.monthly_income:
        return str(understanding.monthly_income)
    if field is InterviewField.EMPLOYMENT_TYPE and understanding.employment_type:
        return understanding.employment_type.value
    if (
        field is InterviewField.MONTHLY_EXPENSES
        and understanding.monthly_expenses is not None
    ):
        return str(understanding.monthly_expenses)
    if field is InterviewField.DEPENDENTS and understanding.dependents is not None:
        return str(understanding.dependents)
    if (
        field is InterviewField.ACTIVE_DEBTS
        and understanding.has_active_debts is not None
    ):
        return "sim" if understanding.has_active_debts else "não"
    return None


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


# Respostas validas de dividas ativas, que nunca podem ser lidas como recusa.
_DEBT_ANSWERS = frozenset({"sim", "nao"})


def _cancels_interview(user_text: str) -> bool:
    """Detecta pedido de cancelamento em qualquer ponto da entrevista.

    Um "nao" isolado responde a pergunta de dividas ativas e jamais cancela.
    Fora isso a deteccao pode ser generosa: os campos so aceitam numero, tipo
    de emprego ou sim/nao, entao nenhuma palavra de recusa colide com resposta
    valida.
    """
    if normalize_short_answer(user_text) in _DEBT_ANSWERS:
        return False
    return detects_refusal(user_text)


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
