"""Testes do repositorio CSV de clientes."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from banco_agil.domain.exceptions import RepositoryError
from banco_agil.repositories.client_csv import ClientCsvRepository, normalize_cpf

CSV_HEADER = "cpf,data_nascimento,limite_credito,score_credito\n"
CSV_ROWS = (
    "\n".join(
        (
            "01234567890,1990-05-20,2500.00,700",
            "98765432100,1985-10-12,4200.50,820",
        )
    )
    + "\n"
)


@pytest.fixture
def clients_path(tmp_path: Path) -> Path:
    """Cria um CSV ficticio isolado para cada teste."""
    path = tmp_path / "clientes.csv"
    path.write_text(CSV_HEADER + CSV_ROWS, encoding="utf-8")
    return path


def test_normalize_cpf_preserves_leading_zero() -> None:
    assert normalize_cpf("012.345.678-90") == "01234567890"


@pytest.mark.parametrize("cpf", ["123", "0123456789A", "012/345/67890"])
def test_normalize_cpf_rejects_invalid_value(cpf: str) -> None:
    with pytest.raises(ValueError, match="CPF"):
        normalize_cpf(cpf)


def test_find_by_cpf_returns_validated_client(clients_path: Path) -> None:
    repository = ClientCsvRepository(clients_path)

    client = repository.find_by_cpf("012.345.678-90")

    assert client is not None
    assert client.cpf == "01234567890"
    assert str(client.credit_limit) == "2500.00"
    assert client.credit_score == 700


def test_find_by_cpf_returns_none_for_unknown_client(clients_path: Path) -> None:
    repository = ClientCsvRepository(clients_path)

    assert repository.find_by_cpf("11111111111") is None


def test_update_credit_score_replaces_same_client_row(clients_path: Path) -> None:
    repository = ClientCsvRepository(clients_path)

    updated = repository.update_credit_score("01234567890", 760)

    assert updated.credit_score == 760
    assert repository.find_by_cpf("01234567890") == updated
    contents = clients_path.read_text(encoding="utf-8")
    assert contents.count("01234567890") == 1
    assert ",2500.00,760" in contents


def test_update_rejects_invalid_score_without_changing_file(
    clients_path: Path,
) -> None:
    original = clients_path.read_bytes()
    repository = ClientCsvRepository(clients_path)

    with pytest.raises(ValidationError):
        repository.update_credit_score("01234567890", 1001)

    assert clients_path.read_bytes() == original


def test_update_rejects_unknown_client(clients_path: Path) -> None:
    repository = ClientCsvRepository(clients_path)

    with pytest.raises(RepositoryError, match="not found"):
        repository.update_credit_score("11111111111", 700)


@pytest.mark.parametrize(
    "contents",
    [
        "",
        CSV_HEADER,
        CSV_HEADER + CSV_ROWS + "01234567890,2000-01-01,100.00,500\n",
        CSV_HEADER + "123,1990-05-20,2500.00,700\n",
        CSV_HEADER + "01234567890,1990-05-20,2500.00,700,extra\n",
        "cpf,data_nascimento,score_credito\n01234567890,1990-05-20,700\n",
    ],
)
def test_invalid_csv_raises_controlled_error(tmp_path: Path, contents: str) -> None:
    path = tmp_path / "clientes.csv"
    path.write_text(contents, encoding="utf-8")
    repository = ClientCsvRepository(path)

    with pytest.raises(RepositoryError, match="invalid clients CSV"):
        repository.find_by_cpf("01234567890")


def test_missing_csv_raises_controlled_error(tmp_path: Path) -> None:
    repository = ClientCsvRepository(tmp_path / "clientes.csv")

    with pytest.raises(RepositoryError, match="could not be read"):
        repository.find_by_cpf("01234567890")


def test_failed_replace_preserves_original_file(
    clients_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = clients_path.read_bytes()
    repository = ClientCsvRepository(clients_path)

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr("banco_agil.repositories.client_csv.os.replace", fail_replace)

    with pytest.raises(RepositoryError, match="could not be updated"):
        repository.update_credit_score("01234567890", 760)

    assert clients_path.read_bytes() == original
    assert list(clients_path.parent.glob("*.tmp")) == []
