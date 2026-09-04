"""Testes de composicao e isolamento dos prompts de runtime."""

from datetime import date
from decimal import Decimal

import pytest
from langchain_core.messages import SystemMessage

from banco_agil.agents.state import ConversationState, CreditInterviewDraft
from banco_agil.domain.enums import Agent, EmploymentType, Intent
from banco_agil.domain.models import Client
from banco_agil.prompts.registry import (
    PROMPT_REGISTRY,
    WELCOME_PROMPT_DEFINITION,
    PromptDefinition,
)
from banco_agil.prompts.renderer import compact_state, render_prompt

EXPECTED_TOOLS = {
    Agent.TRIAGE: {"validate_client_cpf", "authenticate_client", "end_service"},
    Agent.CREDIT: {
        "get_credit_limit",
        "request_limit_increase",
        "end_service",
    },
    Agent.CREDIT_INTERVIEW: {"update_credit_score", "end_service"},
    Agent.EXCHANGE: {"get_exchange_rate", "end_service"},
}


def test_registry_uses_documented_ids_versions_variables_and_limits() -> None:
    assert PROMPT_REGISTRY.global_prompt.prompt_id == "global"
    assert PROMPT_REGISTRY.global_prompt.version == "1.4.0"
    assert PROMPT_REGISTRY.global_prompt.character_limit == 1_200
    assert PROMPT_REGISTRY.global_prompt.variables == frozenset()

    assert WELCOME_PROMPT_DEFINITION.prompt_id == "welcome"
    assert WELCOME_PROMPT_DEFINITION.version == "1.1.0"
    assert WELCOME_PROMPT_DEFINITION.variables == frozenset()
    assert WELCOME_PROMPT_DEFINITION.character_limit == 800

    for agent in Agent:
        definition = PROMPT_REGISTRY.for_agent(agent)
        assert definition.prompt_id == agent.value
        if agent is Agent.TRIAGE:
            expected_version = "1.6.0"
        elif agent is Agent.EXCHANGE:
            expected_version = "1.5.0"
        else:
            expected_version = "1.4.0"
        assert definition.version == expected_version
        assert definition.character_limit == 1_000
        assert definition.variables == frozenset({"state"})


def test_three_llm_specialists_require_natural_contextual_replies() -> None:
    for agent in (Agent.CREDIT, Agent.CREDIT_INTERVIEW, Agent.EXCHANGE):
        prompt = PROMPT_REGISTRY.for_agent(agent).template.casefold()
        assert "natural" in prompt
        assert "preserve" in prompt


def test_prompt_definition_rejects_undeclared_template_variable() -> None:
    with pytest.raises(ValueError, match="template variables"):
        PromptDefinition(
            prompt_id="invalid",
            version="1.0.0",
            template="Estado: {{ unexpected }}",
            variables=frozenset({"state"}),
            character_limit=100,
        )


@pytest.mark.parametrize("agent", list(Agent))
def test_rendering_produces_one_bounded_system_message_for_active_agent(
    agent: Agent,
) -> None:
    rendered = render_prompt(ConversationState(active_agent=agent))
    specialist = PROMPT_REGISTRY.for_agent(agent)

    assert isinstance(rendered.system_message, SystemMessage)
    if agent is Agent.TRIAGE:
        specialist_version = "1.6.0"
    elif agent is Agent.EXCHANGE:
        specialist_version = "1.5.0"
    else:
        specialist_version = "1.4.0"
    assert rendered.prompt_version == (
        f"global@1.4.0+{agent.value}@{specialist_version}"
    )
    assert "{{" not in str(rendered.system_message.content)
    assert len(PROMPT_REGISTRY.global_prompt.template) <= 1_200
    assert len(specialist.template) <= 1_000
    assert (
        len(PROMPT_REGISTRY.global_prompt.template) + len(specialist.template) < 2_200
    )

    for inactive_agent in Agent:
        marker = PROMPT_REGISTRY.for_agent(inactive_agent).template.splitlines()[0]
        if inactive_agent is agent:
            assert marker in str(rendered.system_message.content)
        else:
            assert marker not in str(rendered.system_message.content)


def test_compact_state_omits_pii_financial_values_and_null_fields() -> None:
    state = ConversationState(
        active_agent=Agent.CREDIT_INTERVIEW,
        intent=Intent.LIMIT_INCREASE,
        pending_cpf="01234567890",
        pending_birth_date=date(1990, 5, 20),
        authenticated_client=Client(
            cpf="01234567890",
            birth_date=date(1990, 5, 20),
            credit_limit=Decimal("2500.00"),
            credit_score=700,
        ),
        requested_limit=Decimal("4000.00"),
        interview_draft=CreditInterviewDraft(
            consent_given=True,
            monthly_income=Decimal("9876.54"),
            employment_type=EmploymentType.FORMAL,
            monthly_expenses=Decimal("1234.56"),
            dependents=2,
            has_active_debts=True,
        ),
    )

    compact = compact_state(state)
    rendered = render_prompt(state)
    content = str(rendered.system_message.content)

    assert compact == {
        "is_authenticated": True,
        "interview_authorized": True,
        "interview_step": "complete",
        "collected_fields": [
            "monthly_income",
            "employment_type",
            "monthly_expenses",
            "dependents",
            "has_active_debts",
        ],
    }
    assert len(rendered.compact_state_json) < 500
    assert "01234567890" not in content
    assert "1990-05-20" not in content
    assert "9876.54" not in content
    assert "1234.56" not in content
    assert "2500.00" not in content
    assert "4000.00" not in content


def test_triage_compact_state_exposes_only_current_step_and_safe_context() -> None:
    state = ConversationState(
        authentication_attempts=1,
        intent=Intent.UNKNOWN,
        pending_cpf="01234567890",
    )

    assert compact_state(state) == {
        "is_authenticated": False,
        "authentication_attempts": 1,
        "triage_step": "birth_date",
        "current_intent": "unknown",
    }


@pytest.mark.parametrize("agent", list(Agent))
def test_only_active_specialist_tools_are_exposed(agent: Agent) -> None:
    rendered = render_prompt(ConversationState(active_agent=agent))

    assert {tool.name for tool in rendered.tools} == EXPECTED_TOOLS[agent]
