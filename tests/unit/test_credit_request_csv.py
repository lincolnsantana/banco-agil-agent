"""Testes do repositorio CSV de solicitacoes de credito."""

import csv
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from banco_agil.domain.enums import CreditRequestStatus
from banco_agil.domain.exceptions import RepositoryError
from banco_agil.domain.models import CreditRequest
from banco_agil.repositories.credit_request_csv import CreditRequestCsvRepository

EXPECTED_FIELDS = [
    "cpf_cliente",
    "data_hora_solicitacao",
    "limite_atual",
    "novo_limite_solicitado",
    "status_pedido",
]


@pytest.fixture
def pending_request() -> CreditRequest:
    """Retorna uma solicitacao ficticia pendente."""
    return CreditRequest(
        client_cpf="01234567890",
        requested_at=datetime(2026, 9, 2, 12, 30, tzinfo=UTC),
        current_limit=Decimal("2500.005"),
        requested_limit=Decimal("5000.004"),
        status=CreditRequestStatus.PENDING,
    )


def test_create_builds_file_with_schema_and_two_decimal_places(
    tmp_path: Path,
    pending_request: CreditRequest,
) -> None:
    path = tmp_path / "solicitacoes.csv"
    repository = CreditRequestCsvRepository(path)

    created = repository.create(pending_request)

    with path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
    assert reader.fieldnames == EXPECTED_FIELDS
    assert rows == [
        {
            "cpf_cliente": "01234567890",
            "data_hora_solicitacao": "2026-09-02T12:30:00+00:00",
            "limite_atual": "2500.00",
            "novo_limite_solicitado": "5000.00",
            "status_pedido": "pendente",
        }
    ]
    assert created.current_limit == Decimal("2500.00")


@pytest.mark.parametrize(
    "status",
    [CreditRequestStatus.APPROVED, CreditRequestStatus.REJECTED],
)
def test_finalize_updates_same_row_without_duplication(
    tmp_path: Path,
    pending_request: CreditRequest,
    status: CreditRequestStatus,
) -> None:
    repository = CreditRequestCsvRepository(tmp_path / "solicitacoes.csv")
    created = repository.create(pending_request)

    finalized = repository.finalize(created, status)

    requests = repository.list_all()
    assert finalized.status is status
    assert requests == [finalized]


def test_duplicate_request_is_rejected(
    tmp_path: Path,
    pending_request: CreditRequest,
) -> None:
    repository = CreditRequestCsvRepository(tmp_path / "solicitacoes.csv")
    repository.create(pending_request)

    with pytest.raises(RepositoryError, match="already exists"):
        repository.create(pending_request)


def test_only_pending_requests_can_be_created(
    tmp_path: Path,
    pending_request: CreditRequest,
) -> None:
    repository = CreditRequestCsvRepository(tmp_path / "solicitacoes.csv")
    approved = pending_request.model_copy(
        update={"status": CreditRequestStatus.APPROVED}
    )

    with pytest.raises(ValueError, match="pending"):
        repository.create(approved)


def test_finalize_rejects_non_final_status(
    tmp_path: Path,
    pending_request: CreditRequest,
) -> None:
    repository = CreditRequestCsvRepository(tmp_path / "solicitacoes.csv")
    created = repository.create(pending_request)

    with pytest.raises(ValueError, match="approved or rejected"):
        repository.finalize(created, CreditRequestStatus.PENDING)


def test_invalid_persisted_status_is_controlled(tmp_path: Path) -> None:
    path = tmp_path / "solicitacoes.csv"
    path.write_text(
        ",".join(EXPECTED_FIELDS)
        + "\n01234567890,2026-09-02T12:30:00+00:00,2500.00,5000.00,reprovado\n",
        encoding="utf-8",
    )
    repository = CreditRequestCsvRepository(path)

    with pytest.raises(RepositoryError, match="invalid credit requests CSV"):
        repository.list_all()


def test_failed_replace_preserves_original_requests_file(
    tmp_path: Path,
    pending_request: CreditRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "solicitacoes.csv"
    original = ",".join(EXPECTED_FIELDS) + "\n"
    path.write_text(original, encoding="utf-8")
    repository = CreditRequestCsvRepository(path)

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(
        "banco_agil.repositories.credit_request_csv.os.replace",
        fail_replace,
    )

    with pytest.raises(RepositoryError, match="could not be updated"):
        repository.create(pending_request)

    assert path.read_text(encoding="utf-8") == original
    assert list(tmp_path.glob("*.tmp")) == []
