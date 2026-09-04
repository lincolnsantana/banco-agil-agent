"""No deterministico da entrevista de credito."""

from banco_agil.agents._shared import (
    authentication_reply_if_missing,
    end_reply_if_requested,
    parse_confirmation,
)
from banco_agil.agents.state import ConversationState
from banco_agil.domain.enums import Agent, Intent
from banco_agil.domain.exceptions import DomainError, RepositoryError
from banco_agil.domain.models import ScoreUpdateResult
from banco_agil.services.credit_interview import (
    CreditInterviewService,
    InterviewField,
)
from banco_agil.tools.banking import update_credit_score

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

    if not state.interview_draft.consent_given:
        consent = parse_confirmation(user_text)
        if consent is None:
            return "Deseja realizar a entrevista de crédito? Responda sim ou não."
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
        return _question(progress.next_field)

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
            f"{new_score} com os dados atualizados. Quer que eu analise um "
            "novo limite ou consulte uma moeda?"
        )
    if new_score < previous_score:
        return (
            f"Seu score foi atualizado de {previous_score} para {new_score}, "
            "uma queda pelos dados informados — e isso não define seus "
            "próximos passos. Quer revisar outro serviço ou tentar uma nova "
            "análise de limite?"
        )
    return (
        f"Seu score permanece em {new_score} após a entrevista. Quer "
        "analisar um limite ou consultar uma moeda?"
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


def _question(field: InterviewField | None) -> str:
    if field is None:
        raise ValueError("interview progress must provide the next field")
    return _QUESTIONS[field]
