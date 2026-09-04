"""Registro dos prompts e tools permitidos por especialista."""

import re
from collections.abc import Mapping
from dataclasses import dataclass

from langchain_core.tools import BaseTool

from banco_agil.domain.enums import Agent
from banco_agil.prompts.templates import (
    CREDIT_INTERVIEW_PROMPT,
    CREDIT_PROMPT,
    EXCHANGE_PROMPT,
    GLOBAL_PROMPT,
    TRIAGE_PROMPT,
    WELCOME_PROMPT,
)
from banco_agil.tools.banking import (
    authenticate_client,
    end_service,
    get_credit_limit,
    get_exchange_rate,
    request_limit_increase,
    update_credit_score,
    validate_client_cpf,
)

_VARIABLE_PATTERN = re.compile(r"{{\s*([a-z_]+)\s*}}")
_GLOBAL_PROMPT_VERSION = "1.3.0"
_TRIAGE_PROMPT_VERSION = "1.6.0"
_SPECIALIST_PROMPT_VERSION = "1.3.0"
_EXCHANGE_PROMPT_VERSION = "1.4.0"


@dataclass(frozen=True)
class PromptDefinition:
    """Define um prompt versionado e suas variaveis obrigatorias."""

    prompt_id: str
    version: str
    template: str
    variables: frozenset[str]
    character_limit: int

    def __post_init__(self) -> None:
        """Valida tamanho e contrato de variaveis ao carregar o registro."""
        if len(self.template) > self.character_limit:
            raise ValueError(f"prompt {self.prompt_id} exceeds character limit")
        found_variables = frozenset(_VARIABLE_PATTERN.findall(self.template))
        if found_variables != self.variables:
            raise ValueError(f"prompt {self.prompt_id} has invalid template variables")

    def render(self, **variables: str) -> str:
        """Substitui somente o conjunto declarado de variaveis."""
        if frozenset(variables) != self.variables:
            raise ValueError(f"prompt {self.prompt_id} received invalid variables")
        rendered = self.template
        for name, value in variables.items():
            rendered = re.sub(r"{{\s*" + re.escape(name) + r"\s*}}", value, rendered)
        return rendered


class PromptRegistry:
    """Relaciona cada especialista ao proprio prompt e conjunto de tools."""

    def __init__(
        self,
        global_prompt: PromptDefinition,
        specialist_prompts: Mapping[Agent, PromptDefinition],
        agent_tools: Mapping[Agent, tuple[BaseTool, ...]],
    ) -> None:
        """Valida que todos os especialistas possuem prompt e tools."""
        expected_agents = set(Agent)
        if set(specialist_prompts) != expected_agents:
            raise ValueError("prompt registry must cover every agent")
        if set(agent_tools) != expected_agents:
            raise ValueError("tool registry must cover every agent")
        self.global_prompt = global_prompt
        self._specialist_prompts = dict(specialist_prompts)
        self._agent_tools = dict(agent_tools)

    def for_agent(self, agent: Agent) -> PromptDefinition:
        """Retorna a definicao do especialista ativo."""
        return self._specialist_prompts[agent]

    def tools_for_agent(self, agent: Agent) -> tuple[BaseTool, ...]:
        """Retorna somente as tools permitidas ao especialista ativo."""
        return self._agent_tools[agent]


WELCOME_PROMPT_DEFINITION = PromptDefinition(
    prompt_id="welcome",
    version="1.1.0",
    template=WELCOME_PROMPT,
    variables=frozenset(),
    character_limit=800,
)


PROMPT_REGISTRY = PromptRegistry(
    global_prompt=PromptDefinition(
        prompt_id="global",
        version=_GLOBAL_PROMPT_VERSION,
        template=GLOBAL_PROMPT,
        variables=frozenset(),
        character_limit=1_200,
    ),
    specialist_prompts={
        Agent.TRIAGE: PromptDefinition(
            prompt_id="triage",
            version=_TRIAGE_PROMPT_VERSION,
            template=TRIAGE_PROMPT,
            variables=frozenset({"state"}),
            character_limit=1_000,
        ),
        Agent.CREDIT: PromptDefinition(
            prompt_id="credit",
            version=_SPECIALIST_PROMPT_VERSION,
            template=CREDIT_PROMPT,
            variables=frozenset({"state"}),
            character_limit=1_000,
        ),
        Agent.CREDIT_INTERVIEW: PromptDefinition(
            prompt_id="credit_interview",
            version=_SPECIALIST_PROMPT_VERSION,
            template=CREDIT_INTERVIEW_PROMPT,
            variables=frozenset({"state"}),
            character_limit=1_000,
        ),
        Agent.EXCHANGE: PromptDefinition(
            prompt_id="exchange",
            version=_EXCHANGE_PROMPT_VERSION,
            template=EXCHANGE_PROMPT,
            variables=frozenset({"state"}),
            character_limit=1_000,
        ),
    },
    agent_tools={
        Agent.TRIAGE: (validate_client_cpf, authenticate_client, end_service),
        Agent.CREDIT: (get_credit_limit, request_limit_increase, end_service),
        Agent.CREDIT_INTERVIEW: (update_credit_score, end_service),
        Agent.EXCHANGE: (get_exchange_rate, end_service),
    },
)
