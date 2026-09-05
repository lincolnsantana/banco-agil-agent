"""Renderizacao de prompt com contexto compacto e sem PII."""

import json
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool

from banco_agil.agents.state import ConversationState, CreditInterviewDraft
from banco_agil.domain.enums import Agent
from banco_agil.prompts.registry import (
    PROMPT_REGISTRY,
    UNDERSTANDING_PROMPT_DEFINITION,
    PromptDefinition,
    PromptRegistry,
)

CompactValue = bool | int | str | list[str]
CompactState = dict[str, CompactValue]
_STATE_CHARACTER_LIMIT = 500
_COMBINED_PROMPT_LIMIT = 2_200


@dataclass(frozen=True)
class RenderedPrompt:
    """Agrupa a mensagem unica, versao e tools do especialista ativo."""

    system_message: SystemMessage
    tools: tuple[BaseTool, ...]
    prompt_version: str
    compact_state_json: str


def compact_state(state: ConversationState) -> CompactState:
    """Seleciona apenas contexto necessario e nao sensivel por especialista."""
    compact: CompactState = {"is_authenticated": state.authenticated}
    if state.active_agent is Agent.TRIAGE:
        compact.update(
            {
                "authentication_attempts": state.authentication_attempts,
                "triage_step": _triage_step(state),
                "current_intent": state.intent.value,
            }
        )
    elif state.active_agent is Agent.CREDIT or state.active_agent is Agent.KNOWLEDGE:
        compact["current_intent"] = state.intent.value
    elif state.active_agent is Agent.CREDIT_INTERVIEW:
        draft = state.interview_draft
        compact.update(
            {
                "interview_authorized": draft.consent_given,
                "interview_step": _interview_step(draft),
                "collected_fields": _collected_interview_fields(draft),
            }
        )
    return compact


def render_prompt(
    state: ConversationState,
    registry: PromptRegistry = PROMPT_REGISTRY,
) -> RenderedPrompt:
    """Compoe global, especialista ativo e estado em um system message."""
    specialist = registry.for_agent(state.active_agent)
    static_length = len(registry.global_prompt.template) + len(specialist.template)
    if static_length >= _COMBINED_PROMPT_LIMIT:
        raise ValueError("combined prompt exceeds character limit")

    state_json = json.dumps(
        compact_state(state),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if len(state_json) >= _STATE_CHARACTER_LIMIT:
        raise ValueError("compact state exceeds character limit")

    content = "\n\n".join(
        (
            registry.global_prompt.render(),
            specialist.render(state=state_json),
        )
    )
    return RenderedPrompt(
        system_message=SystemMessage(content=content),
        tools=registry.tools_for_agent(state.active_agent),
        prompt_version=(
            f"{registry.global_prompt.prompt_id}@{registry.global_prompt.version}"
            f"+{specialist.prompt_id}@{specialist.version}"
        ),
        compact_state_json=state_json,
    )


def render_understanding_prompt(
    flow: str,
    topics: Sequence[str],
    registry: PromptRegistry = PROMPT_REGISTRY,
    definition: PromptDefinition = UNDERSTANDING_PROMPT_DEFINITION,
) -> RenderedPrompt:
    """Compoe global e entendimento do turno para o fluxo em andamento.

    Nao envia estado nem tools: a saida e um schema fechado que o Python
    aterra no texto do cliente antes de usar. O fluxo atual e os topicos do
    catalogo chegam como texto curto, sem dado do cliente.
    """
    static_length = len(registry.global_prompt.template) + len(definition.template)
    if static_length >= _COMBINED_PROMPT_LIMIT:
        raise ValueError("combined prompt exceeds character limit")
    content = "\n\n".join(
        (
            registry.global_prompt.render(),
            definition.render(flow=flow, topics=", ".join(topics)),
        )
    )
    return RenderedPrompt(
        system_message=SystemMessage(content=content),
        tools=(),
        prompt_version=(
            f"{registry.global_prompt.prompt_id}@{registry.global_prompt.version}"
            f"+{definition.prompt_id}@{definition.version}"
        ),
        compact_state_json="{}",
    )


def _triage_step(state: ConversationState) -> str:
    if state.authenticated:
        return "identify_intent"
    if state.pending_cpf is None:
        return "cpf"
    if state.pending_birth_date is None:
        return "birth_date"
    return "authenticate"


def _interview_step(draft: CreditInterviewDraft) -> str:
    if not draft.consent_given:
        return "consent"
    for field_name in (
        "monthly_income",
        "employment_type",
        "monthly_expenses",
        "dependents",
        "has_active_debts",
    ):
        if getattr(draft, field_name) is None:
            return field_name
    return "complete"


def _collected_interview_fields(draft: CreditInterviewDraft) -> list[str]:
    return [
        field_name
        for field_name in (
            "monthly_income",
            "employment_type",
            "monthly_expenses",
            "dependents",
            "has_active_debts",
        )
        if getattr(draft, field_name) is not None
    ]
