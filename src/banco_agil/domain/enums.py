"""Enumeracoes compartilhadas pelo dominio e pela orquestracao."""

from enum import StrEnum


class Agent(StrEnum):
    """Especialista interno responsavel pelo turno atual."""

    TRIAGE = "triage"
    CREDIT = "credit"
    CREDIT_INTERVIEW = "credit_interview"
    EXCHANGE = "exchange"


class Intent(StrEnum):
    """Necessidade identificada na conversa."""

    UNKNOWN = "unknown"
    CREDIT_LIMIT = "credit_limit"
    LIMIT_INCREASE = "limit_increase"
    EXCHANGE_RATE = "exchange_rate"
    OTHER = "other"
    END_SERVICE = "end_service"


class CreditRequestStatus(StrEnum):
    """Estado persistivel de uma solicitacao de aumento de limite."""

    PENDING = "pendente"
    APPROVED = "aprovado"
    REJECTED = "rejeitado"


class EmploymentType(StrEnum):
    """Tipos de emprego aceitos na entrevista de credito."""

    FORMAL = "formal"
    SELF_EMPLOYED = "autônomo"
    UNEMPLOYED = "desempregado"


class EndReason(StrEnum):
    """Motivo controlado para encerramento do atendimento."""

    USER_REQUEST = "user_request"
    AUTHENTICATION_FAILURES = "authentication_failures"
    COMPLETED = "completed"


class AuditEventType(StrEnum):
    """Etapa tecnica registrada pela auditoria, sem dados pessoais."""

    STARTED = "started"
    TRANSITION = "transition"
    INTEGRATION = "integration"
    ERROR = "error"
    FINISHED = "finished"
    ENDED = "ended"
