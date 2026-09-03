"""Testes offline do adaptador estruturado de LLM."""

import ast
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, SecretStr, ValidationError

from banco_agil.config import Settings
from banco_agil.domain.enums import Intent
from banco_agil.domain.exceptions import IntegrationError
from banco_agil.integrations.llm import (
    FakeStructuredLlm,
    GroqStructuredLlm,
    LlmCallLimitError,
)
from banco_agil.observability.metrics import InMemoryLlmMetricsRecorder


class IntentOutput(BaseModel):
    """Saida estruturada minima usada pelos testes."""

    intent: Intent


class FakeRunnable:
    """Runnable controlado retornado pelo falso ChatGroq."""

    response: ClassVar[object]
    error: ClassVar[Exception | None] = None
    messages: ClassVar[Sequence[BaseMessage] | None] = None

    def invoke(self, messages: Sequence[BaseMessage]) -> object:
        """Registra mensagens e retorna ou levanta o resultado configurado."""
        self.__class__.messages = messages
        if self.error is not None:
            raise self.error
        return self.response


class FakeChatGroq:
    """Substituto offline que captura a configuracao do provedor."""

    init_kwargs: ClassVar[dict[str, object]] = {}
    schema: ClassVar[type[BaseModel] | None] = None
    include_raw: ClassVar[bool | None] = None

    def __init__(self, **kwargs: object) -> None:
        self.__class__.init_kwargs = kwargs

    def with_structured_output(
        self,
        schema: type[BaseModel],
        *,
        include_raw: bool,
    ) -> FakeRunnable:
        """Captura o schema e devolve o runnable controlado."""
        self.__class__.schema = schema
        self.__class__.include_raw = include_raw
        return FakeRunnable()


@pytest.fixture(autouse=True)
def reset_fake_provider() -> None:
    """Isola estado compartilhado do provedor falso."""
    FakeChatGroq.init_kwargs = {}
    FakeChatGroq.schema = None
    FakeChatGroq.include_raw = None
    FakeRunnable.error = None
    FakeRunnable.messages = None
    FakeRunnable.response = {
        "parsed": {"intent": "credit_limit"},
        "raw": AIMessage(
            content="",
            usage_metadata={
                "input_tokens": 12,
                "output_tokens": 3,
                "total_tokens": 15,
            },
        ),
        "parsing_error": None,
    }


def build_settings() -> Settings:
    """Cria configuracao ficticia sem consultar arquivo de ambiente."""
    return Settings(
        _env_file=None,
        groq_api_key=SecretStr("test-key"),
    )


def test_settings_use_required_llm_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.groq_api_key is None
    assert settings.groq_model == "llama-3.3-70b-versatile"
    assert settings.llm_temperature == 0.1
    assert settings.llm_max_tokens == 180
    assert settings.llm_timeout_seconds == 20.0


def test_settings_hide_configured_api_key() -> None:
    settings = build_settings()

    assert "test-key" not in repr(settings)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("llm_temperature", 1.1),
        ("llm_max_tokens", 0),
        ("llm_timeout_seconds", 0),
    ],
)
def test_settings_reject_invalid_llm_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate({field: value})


def test_fake_llm_is_deterministic_and_validates_schema() -> None:
    fake = FakeStructuredLlm({"intent": "exchange_rate"})
    messages = [HumanMessage(content="Quero saber a cotação")]

    first = fake.invoke_structured("turn-1", messages, IntentOutput)
    second = fake.invoke_structured("turn-2", messages, IntentOutput)

    assert first == IntentOutput(intent=Intent.EXCHANGE_RATE)
    assert second == first


def test_fake_llm_rejects_invalid_structured_output() -> None:
    fake = FakeStructuredLlm({"intent": "unsupported"})

    with pytest.raises(IntegrationError, match="invalid structured LLM output"):
        fake.invoke_structured("turn-1", [], IntentOutput)


def test_fake_llm_blocks_second_call_in_same_turn() -> None:
    fake = FakeStructuredLlm({"intent": "other"})
    fake.invoke_structured("turn-1", [], IntentOutput)

    with pytest.raises(LlmCallLimitError):
        fake.invoke_structured("turn-1", [], IntentOutput)


def test_groq_adapter_applies_configuration_and_records_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("banco_agil.integrations.llm.ChatGroq", FakeChatGroq)
    timer_values = iter([10.0, 10.25])
    monkeypatch.setattr(
        "banco_agil.integrations.llm.perf_counter",
        lambda: next(timer_values),
    )
    recorder = InMemoryLlmMetricsRecorder()
    adapter = GroqStructuredLlm(build_settings(), metrics_recorder=recorder)
    messages = [HumanMessage(content="Preciso de ajuda")]

    result = adapter.invoke_structured(
        "turn-1",
        messages,
        IntentOutput,
        prompt_version="triage:1.1.0",
    )

    assert result.intent is Intent.CREDIT_LIMIT
    assert FakeChatGroq.init_kwargs == {
        "api_key": SecretStr("test-key"),
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.1,
        "max_tokens": 180,
        "timeout": 20.0,
        "max_retries": 0,
    }
    assert FakeChatGroq.schema is IntentOutput
    assert FakeChatGroq.include_raw is True
    assert FakeRunnable.messages == messages
    assert len(recorder.calls) == 1
    assert recorder.calls[0].duration_ms == 250.0
    assert recorder.calls[0].input_tokens == 12
    assert recorder.calls[0].output_tokens == 3
    assert recorder.calls[0].prompt_version == "triage:1.1.0"
    assert recorder.calls[0].succeeded


def test_groq_adapter_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("banco_agil.integrations.llm.ChatGroq", FakeChatGroq)

    with pytest.raises(ValueError, match="API key"):
        GroqStructuredLlm(Settings(_env_file=None))


def test_timeout_is_controlled_recorded_and_consumes_turn_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("banco_agil.integrations.llm.ChatGroq", FakeChatGroq)
    FakeRunnable.error = TimeoutError("simulated timeout")
    timer_values = iter([1.0, 1.2])
    monkeypatch.setattr(
        "banco_agil.integrations.llm.perf_counter",
        lambda: next(timer_values),
    )
    recorder = InMemoryLlmMetricsRecorder()
    adapter = GroqStructuredLlm(build_settings(), metrics_recorder=recorder)

    with pytest.raises(IntegrationError, match="LLM provider failed"):
        adapter.invoke_structured("turn-1", [], IntentOutput)
    with pytest.raises(LlmCallLimitError):
        adapter.invoke_structured("turn-1", [], IntentOutput)

    assert len(recorder.calls) == 1
    assert recorder.calls[0].duration_ms == pytest.approx(200.0)
    assert not recorder.calls[0].succeeded
    assert recorder.calls[0].input_tokens is None


def test_invalid_provider_output_is_controlled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("banco_agil.integrations.llm.ChatGroq", FakeChatGroq)
    FakeRunnable.response = {
        "parsed": {"intent": "invalid"},
        "raw": AIMessage(content=""),
        "parsing_error": None,
    }
    adapter = GroqStructuredLlm(build_settings())

    with pytest.raises(IntegrationError, match="invalid structured LLM output"):
        adapter.invoke_structured("turn-1", [], IntentOutput)


def test_provider_sdk_is_imported_only_by_llm_adapter() -> None:
    source_root = Path(__file__).parents[2] / "src" / "banco_agil"
    offenders: list[Path] = []

    for source_path in source_root.rglob("*.py"):
        if source_path.name == "llm.py":
            continue
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module_names: list[str] = []
            if isinstance(node, ast.Import):
                module_names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                module_names = [node.module]
            if any(
                name.split(".")[0] in {"groq", "langchain_groq"}
                for name in module_names
            ):
                offenders.append(source_path)

    assert offenders == []
