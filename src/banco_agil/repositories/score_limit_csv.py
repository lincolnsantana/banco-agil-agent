"""Consulta validada das faixas de score e limite em CSV."""

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from banco_agil.domain.exceptions import RepositoryError

SCORE_LIMIT_FIELDS = ("score_minimo", "score_maximo", "limite_maximo")


@dataclass(frozen=True)
class _ScoreLimitRange:
    minimum_score: int
    maximum_score: int
    maximum_limit: Decimal


class ScoreLimitCsvRepository:
    """Seleciona o limite da faixa inclusiva correspondente ao score."""

    def __init__(self, path: Path) -> None:
        """Configura o caminho do arquivo de faixas."""
        self._path = path

    def find_max_limit(self, score: int) -> Decimal:
        """Retorna o limite permitido para um score entre zero e mil.

        Raises:
            ValueError: Se o score estiver fora do intervalo aceito.
            RepositoryError: Se o CSV estiver ausente ou invalido.
        """
        if (
            isinstance(score, bool)
            or not isinstance(score, int)
            or not 0 <= score <= 1000
        ):
            raise ValueError("score must be an integer between 0 and 1000")

        ranges = self._read_ranges()
        return next(
            score_range.maximum_limit
            for score_range in ranges
            if score_range.minimum_score <= score <= score_range.maximum_score
        )

    def _read_ranges(self) -> list[_ScoreLimitRange]:
        try:
            with self._path.open(encoding="utf-8", newline="") as csv_file:
                reader = csv.DictReader(csv_file, strict=True)
                if tuple(reader.fieldnames or ()) != SCORE_LIMIT_FIELDS:
                    raise RepositoryError(
                        "invalid score limits CSV: unexpected columns"
                    )
                rows = list(reader)
        except OSError as error:
            raise RepositoryError("score limits CSV could not be read") from error
        except csv.Error as error:
            raise RepositoryError("invalid score limits CSV") from error

        try:
            ranges = [self._parse_range(row) for row in rows]
            self._validate_coverage(ranges)
        except (InvalidOperation, KeyError, TypeError, ValueError) as error:
            raise RepositoryError("invalid score limits CSV") from error
        return ranges

    @staticmethod
    def _parse_range(row: dict[str | None, str | list[str] | None]) -> _ScoreLimitRange:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("unexpected row columns")
        score_range = _ScoreLimitRange(
            minimum_score=int(_required_cell(row, "score_minimo")),
            maximum_score=int(_required_cell(row, "score_maximo")),
            maximum_limit=Decimal(_required_cell(row, "limite_maximo")),
        )
        if (
            score_range.minimum_score < 0
            or score_range.maximum_score > 1000
            or score_range.minimum_score > score_range.maximum_score
            or not score_range.maximum_limit.is_finite()
            or score_range.maximum_limit < 0
        ):
            raise ValueError("invalid score range")
        return score_range

    @staticmethod
    def _validate_coverage(ranges: list[_ScoreLimitRange]) -> None:
        if not ranges:
            raise ValueError("score ranges cannot be empty")
        ranges.sort(key=lambda score_range: score_range.minimum_score)
        if ranges[0].minimum_score != 0 or ranges[-1].maximum_score != 1000:
            raise ValueError("score ranges must cover 0 through 1000")
        for previous, current in zip(ranges, ranges[1:], strict=False):
            if current.minimum_score != previous.maximum_score + 1:
                raise ValueError("score ranges cannot overlap or have gaps")


def _required_cell(
    row: dict[str | None, str | list[str] | None],
    field: str,
) -> str:
    value = row.get(field)
    if not isinstance(value, str):
        raise ValueError("invalid CSV cell")
    return value
