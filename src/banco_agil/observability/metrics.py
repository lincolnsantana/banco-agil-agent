"""Contratos minimos para metricas de chamadas ao LLM."""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class LlmCallMetrics:
    """Metricas tecnicas de uma chamada, sem prompt ou dados pessoais."""

    model: str
    duration_ms: float
    succeeded: bool
    prompt_version: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class LlmMetricsRecorder(Protocol):
    """Contrato para receber metricas tecnicas do provedor."""

    def record(self, metrics: LlmCallMetrics) -> None:
        """Recebe uma metrica tecnica da chamada."""
        ...


class NullLlmMetricsRecorder:
    """Descarta metricas quando nenhum coletor foi configurado."""

    def record(self, metrics: LlmCallMetrics) -> None:
        """Aceita a metrica sem efeito colateral."""


@dataclass
class InMemoryLlmMetricsRecorder:
    """Armazena metricas em memoria para testes offline."""

    calls: list[LlmCallMetrics] = field(default_factory=list)

    def record(self, metrics: LlmCallMetrics) -> None:
        """Adiciona uma metrica a colecao observavel."""
        self.calls.append(metrics)
