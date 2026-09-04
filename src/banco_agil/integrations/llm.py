"""Adaptador estruturado do Groq com orcamento de duas chamadas por turno."""

from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Protocol, TypeVar, cast

import httpx
from groq import APIError
from langchain_core.messages import AIMessage, BaseMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, ValidationError

from banco_agil.config import Settings
from banco_agil.domain.exceptions import IntegrationError
from banco_agil.observability.metrics import (
    LlmCallMetrics,
    LlmMetricsRecorder,
    NullLlmMetricsRecorder,
)

MAX_CALLS_PER_TURN = 2

OutputModel = TypeVar("OutputModel", bound=BaseModel)


class LlmCallLimitError(IntegrationError):
    """Indica que o turno ja consumiu suas duas chamadas permitidas."""


class StructuredLlm(Protocol):
    """Contrato generico para uma resposta validada por schema Pydantic."""

    def invoke_structured(
        self,
        turn_id: str,
        messages: Sequence[BaseMessage],
        output_schema: type[OutputModel],
        *,
        prompt_version: str | None = None,
    ) -> OutputModel:
        """Executa no maximo duas chamadas por turno e valida a resposta."""
        ...

    def calls_remaining(self, turn_id: str) -> int:
        """Informa quantas chamadas restam no orçamento do turno."""
        ...


class _TurnCallBudget:
    def __init__(self) -> None:
        self._used_calls: dict[str, int] = {}

    def reserve(self, turn_id: str) -> None:
        if not turn_id:
            raise ValueError("turn ID cannot be empty")
        if self._used_calls.get(turn_id, 0) >= MAX_CALLS_PER_TURN:
            raise LlmCallLimitError("LLM call budget already consumed for this turn")
        self._used_calls[turn_id] = self._used_calls.get(turn_id, 0) + 1

    def calls_remaining(self, turn_id: str) -> int:
        """Retorna o saldo de chamadas do turno, sem consumir orçamento."""
        return max(0, MAX_CALLS_PER_TURN - self._used_calls.get(turn_id, 0))


class FakeStructuredLlm:
    """Fake deterministico que valida respostas sem rede ou credencial."""

    def __init__(self, response: object) -> None:
        """Configura a mesma resposta bruta para todos os turnos."""
        self._response = response
        self._budget = _TurnCallBudget()

    def calls_remaining(self, turn_id: str) -> int:
        """Informa o saldo de chamadas deste fake no turno."""
        return self._budget.calls_remaining(turn_id)

    def invoke_structured(
        self,
        turn_id: str,
        messages: Sequence[BaseMessage],
        output_schema: type[OutputModel],
        *,
        prompt_version: str | None = None,
    ) -> OutputModel:
        """Retorna a resposta configurada validada pelo schema solicitado."""
        del messages, prompt_version
        self._budget.reserve(turn_id)
        try:
            return output_schema.model_validate(self._response)
        except ValidationError as error:
            raise IntegrationError("invalid structured LLM output") from error


class GroqStructuredLlm:
    """Implementa saida estruturada do Groq com telemetria segura."""

    def __init__(
        self,
        settings: Settings,
        metrics_recorder: LlmMetricsRecorder | None = None,
    ) -> None:
        """Configura o modelo sem realizar chamada externa."""
        api_key = settings.groq_api_key
        if api_key is None or not api_key.get_secret_value():
            raise ValueError("Groq API key is required")
        self._model_name = settings.groq_model
        self._model = ChatGroq(
            api_key=api_key,
            model=settings.groq_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
            max_retries=0,
        )
        self._metrics_recorder = metrics_recorder or NullLlmMetricsRecorder()
        self._budget = _TurnCallBudget()

    def calls_remaining(self, turn_id: str) -> int:
        """Informa o saldo de chamadas do adaptador no turno."""
        return self._budget.calls_remaining(turn_id)

    def invoke_structured(
        self,
        turn_id: str,
        messages: Sequence[BaseMessage],
        output_schema: type[OutputModel],
        *,
        prompt_version: str | None = None,
    ) -> OutputModel:
        """Invoca o provedor uma vez e valida a saida estruturada.

        Raises:
            LlmCallLimitError: Se o turno ja realizou duas chamadas.
            IntegrationError: Se o provedor ou a validacao falhar.
        """
        self._budget.reserve(turn_id)
        started_at = perf_counter()
        raw_message: AIMessage | None = None
        succeeded = False
        try:
            runnable = self._model.with_structured_output(
                output_schema,
                include_raw=True,
            )
            response: object = runnable.invoke(list(messages))
            parsed, raw_message = _parse_provider_response(response)
            result = output_schema.model_validate(parsed)
            succeeded = True
            return result
        except (ValidationError, ValueError) as error:
            raise IntegrationError("invalid structured LLM output") from error
        except (APIError, httpx.HTTPError, TimeoutError) as error:
            raise IntegrationError("LLM provider failed") from error
        finally:
            input_tokens, output_tokens = _extract_tokens(raw_message)
            self._metrics_recorder.record(
                LlmCallMetrics(
                    model=self._model_name,
                    duration_ms=(perf_counter() - started_at) * 1000,
                    succeeded=succeeded,
                    prompt_version=prompt_version,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            )


def _parse_provider_response(response: object) -> tuple[object, AIMessage | None]:
    if not isinstance(response, dict):
        raise ValueError("structured response must be an object")
    response_mapping = cast(Mapping[object, object], response)
    if response_mapping.get("parsing_error") is not None:
        raise ValueError("provider could not parse structured output")
    parsed = response_mapping.get("parsed")
    if parsed is None:
        raise ValueError("parsed output is missing")
    raw = response_mapping.get("raw")
    return parsed, raw if isinstance(raw, AIMessage) else None


def _extract_tokens(message: AIMessage | None) -> tuple[int | None, int | None]:
    if message is None or message.usage_metadata is None:
        return None, None
    return (
        message.usage_metadata.get("input_tokens"),
        message.usage_metadata.get("output_tokens"),
    )
