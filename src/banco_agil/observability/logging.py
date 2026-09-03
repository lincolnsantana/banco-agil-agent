"""Logs estruturados e seguros, sem dados pessoais ou financeiros."""

import logging
from collections.abc import Mapping

SENSITIVE_LOG_KEYS = frozenset(
    {
        "cpf",
        "birth_date",
        "pending_cpf",
        "pending_birth_date",
        "client",
        "monthly_income",
        "employment_type",
        "monthly_expenses",
        "dependents",
        "has_active_debts",
        "requested_limit",
        "user_text",
        "reply",
        "message",
        "messages",
        "history",
        "prompt",
    }
)


def get_logger(name: str = "banco_agil") -> logging.Logger:
    """Retorna o logger nomeado sem configurar handlers ou rede."""
    return logging.getLogger(name)


def sanitize_context(context: Mapping[str, object]) -> dict[str, object]:
    """Remove chaves sensiveis antes de registrar ou auditar."""
    return {
        key: value for key, value in context.items() if key not in SENSITIVE_LOG_KEYS
    }
