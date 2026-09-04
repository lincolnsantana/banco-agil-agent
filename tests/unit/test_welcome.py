"""Testes da apresentação inicial gerada pelo Groq."""

from dataclasses import dataclass, field
from typing import TypeVar

from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import BaseModel

from banco_agil.domain.exceptions import IntegrationError
from banco_agil.services.welcome import (
    DEFAULT_WELCOME_MESSAGE,
    generate_welcome_message,
)

OutputModel = TypeVar("OutputModel", bound=BaseModel)


@dataclass
class RecordingWelcomeLlm:
    """Registra a chamada isolada de apresentação."""

    response: object
    calls: list[tuple[str, list[BaseMessage], str | None]] = field(default_factory=list)
    should_fail: bool = False

    def calls_remaining(self, turn_id: str) -> int:
        """Mantém orçamento disponível para a apresentação."""
        del turn_id
        return 2

    def invoke_structured(
        self,
        turn_id: str,
        messages: list[BaseMessage],
        output_schema: type[OutputModel],
        *,
        prompt_version: str | None = None,
    ) -> OutputModel:
        """Valida a resposta configurada ou simula falha externa."""
        self.calls.append((turn_id, messages, prompt_version))
        if self.should_fail:
            raise IntegrationError("provider unavailable")
        return output_schema.model_validate(self.response)


def test_welcome_is_generated_from_isolated_versioned_prompt() -> None:
    expected = (
        "Olá! Sou o assistente virtual do Banco Ágil. Posso ajudar com limite de "
        "crédito, aumento, entrevista de crédito e cotação de moedas. A "
        "autenticação vem primeiro: por favor, informe seu CPF com 11 dígitos."
    )
    llm = RecordingWelcomeLlm({"message": expected})

    message = generate_welcome_message(llm, lambda: "fixed-id")

    assert message == expected
    assert len(llm.calls) == 1
    turn_id, messages, version = llm.calls[0]
    assert turn_id == "welcome-fixed-id"
    assert version == "welcome@1.1.0"
    assert len(messages) == 1
    assert isinstance(messages[0], SystemMessage)
    assert "{{" not in str(messages[0].content)


def test_welcome_falls_back_without_llm_or_on_failure() -> None:
    assert generate_welcome_message(None) == DEFAULT_WELCOME_MESSAGE
    failing_llm = RecordingWelcomeLlm({}, should_fail=True)
    assert generate_welcome_message(failing_llm) == DEFAULT_WELCOME_MESSAGE


def test_welcome_rejects_message_that_omits_required_services() -> None:
    llm = RecordingWelcomeLlm(
        {"message": "Olá! Sou o assistente do Banco Ágil. Como posso ajudar você?"}
    )

    assert generate_welcome_message(llm) == DEFAULT_WELCOME_MESSAGE


def test_canonical_welcome_requests_cpf_before_questions() -> None:
    normalized = DEFAULT_WELCOME_MESSAGE.casefold()

    assert "cpf" in normalized
    assert "autentica" in normalized.replace("autenticação", "autentica")
    assert "nascimento" not in normalized


def test_welcome_rejects_message_that_requests_birth_date() -> None:
    llm = RecordingWelcomeLlm(
        {
            "message": (
                "Olá! Sou o assistente virtual do Banco Ágil. Posso ajudar com "
                "limite de crédito, aumento, entrevista de crédito e cotação de "
                "moedas. A autenticação vem primeiro: informe seu CPF e sua data "
                "de nascimento."
            )
        }
    )

    assert generate_welcome_message(llm) == DEFAULT_WELCOME_MESSAGE
