"""Testes do repositorio de faixas de score."""

from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest

from banco_agil.domain.exceptions import RepositoryError
from banco_agil.repositories.score_limit_csv import ScoreLimitCsvRepository

HEADER = "score_minimo,score_maximo,limite_maximo\n"


def test_versioned_ranges_cover_boundaries() -> None:
    path = Path(__file__).parents[2] / "data" / "score_limite.csv"
    repository = ScoreLimitCsvRepository(path)

    assert repository.find_max_limit(0) == Decimal("1000.00")
    assert repository.find_max_limit(299) == Decimal("1000.00")
    assert repository.find_max_limit(300) == Decimal("2500.00")
    assert repository.find_max_limit(1000) == Decimal("20000.00")


@pytest.mark.parametrize("score", [-1, 1001, "500"])
def test_score_outside_domain_is_rejected(tmp_path: Path, score: object) -> None:
    path = tmp_path / "score_limite.csv"
    path.write_text(HEADER + "0,1000,1000.00\n", encoding="utf-8")
    repository = ScoreLimitCsvRepository(path)

    with pytest.raises(ValueError, match="score"):
        repository.find_max_limit(cast(int, score))


@pytest.mark.parametrize(
    "rows",
    [
        "0,499,1000.00\n501,1000,2000.00\n",
        "0,500,1000.00\n500,1000,2000.00\n",
        "1,1000,2000.00\n",
        "0,999,2000.00\n",
        "0,1000,-1.00\n",
        "invalid,1000,2000.00\n",
    ],
)
def test_invalid_ranges_raise_controlled_error(tmp_path: Path, rows: str) -> None:
    path = tmp_path / "score_limite.csv"
    path.write_text(HEADER + rows, encoding="utf-8")
    repository = ScoreLimitCsvRepository(path)

    with pytest.raises(RepositoryError, match="invalid score limits CSV"):
        repository.find_max_limit(500)


def test_missing_score_file_raises_controlled_error(tmp_path: Path) -> None:
    repository = ScoreLimitCsvRepository(tmp_path / "score_limite.csv")

    with pytest.raises(RepositoryError, match="could not be read"):
        repository.find_max_limit(500)
