"""Persistencia atomica de solicitacoes de aumento de limite."""

import csv
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from tempfile import NamedTemporaryFile

from filelock import FileLock
from pydantic import ValidationError

from banco_agil.domain.enums import CreditRequestStatus
from banco_agil.domain.exceptions import RepositoryError
from banco_agil.domain.models import CreditRequest

CREDIT_REQUEST_FIELDS = (
    "cpf_cliente",
    "data_hora_solicitacao",
    "limite_atual",
    "novo_limite_solicitado",
    "status_pedido",
)
TWO_DECIMAL_PLACES = Decimal("0.01")


class CreditRequestCsvRepository:
    """Cria e finaliza solicitacoes na mesma linha do CSV."""

    def __init__(self, path: Path) -> None:
        """Configura o arquivo mutavel e seu lock lateral."""
        self._path = path
        self._lock = FileLock(f"{path}.lock")

    def create(self, request: CreditRequest) -> CreditRequest:
        """Persiste uma solicitacao pendente com valores normalizados.

        Raises:
            ValueError: Se a solicitacao nao estiver pendente.
            RepositoryError: Se ela for duplicada ou a escrita falhar.
        """
        if request.status is not CreditRequestStatus.PENDING:
            raise ValueError("new credit request must be pending")
        normalized = self._normalize_request(request, CreditRequestStatus.PENDING)

        with self._locked():
            requests = self._read_requests()
            if any(
                self._identity(item) == self._identity(normalized) for item in requests
            ):
                raise RepositoryError("credit request already exists")
            requests.append(normalized)
            self._write_requests(requests)
        return normalized

    def finalize(
        self,
        request: CreditRequest,
        status: CreditRequestStatus,
    ) -> CreditRequest:
        """Atualiza a solicitacao pendente existente sem criar outra linha.

        Raises:
            ValueError: Se o status nao for aprovado ou rejeitado.
            RepositoryError: Se a solicitacao nao existir ou ja estiver finalizada.
        """
        if status not in (
            CreditRequestStatus.APPROVED,
            CreditRequestStatus.REJECTED,
        ):
            raise ValueError("final status must be approved or rejected")
        normalized = self._normalize_request(request, request.status)

        with self._locked():
            requests = self._read_requests()
            request_index = next(
                (
                    index
                    for index, current in enumerate(requests)
                    if self._identity(current) == self._identity(normalized)
                ),
                None,
            )
            if request_index is None:
                raise RepositoryError("credit request not found")
            if requests[request_index].status is not CreditRequestStatus.PENDING:
                raise RepositoryError("credit request is not pending")

            finalized = self._normalize_request(requests[request_index], status)
            requests[request_index] = finalized
            self._write_requests(requests)
        return finalized

    def list_all(self) -> list[CreditRequest]:
        """Retorna todas as solicitacoes validadas na ordem persistida."""
        with self._locked():
            return self._read_requests()

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._lock:
            yield

    def _read_requests(self) -> list[CreditRequest]:
        try:
            with self._path.open(encoding="utf-8", newline="") as csv_file:
                reader = csv.DictReader(csv_file, strict=True)
                if reader.fieldnames is None:
                    return []
                if tuple(reader.fieldnames) != CREDIT_REQUEST_FIELDS:
                    raise RepositoryError(
                        "invalid credit requests CSV: unexpected columns"
                    )
                rows = list(reader)
        except FileNotFoundError:
            return []
        except OSError as error:
            raise RepositoryError("credit requests CSV could not be read") from error
        except csv.Error as error:
            raise RepositoryError("invalid credit requests CSV") from error

        requests: list[CreditRequest] = []
        identities: set[tuple[str, datetime, Decimal, Decimal]] = set()
        try:
            for row in rows:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("unexpected row columns")
                request = CreditRequest.model_validate(
                    {
                        "client_cpf": row["cpf_cliente"],
                        "requested_at": row["data_hora_solicitacao"],
                        "current_limit": row["limite_atual"],
                        "requested_limit": row["novo_limite_solicitado"],
                        "status": row["status_pedido"],
                    }
                )
                identity = self._identity(request)
                if identity in identities:
                    raise ValueError("duplicate credit request")
                identities.add(identity)
                requests.append(request)
        except (KeyError, TypeError, ValueError, ValidationError) as error:
            raise RepositoryError("invalid credit requests CSV") from error
        return requests

    def _write_requests(self, requests: list[CreditRequest]) -> None:
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=self._path.parent,
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                writer = csv.DictWriter(
                    temporary_file,
                    fieldnames=CREDIT_REQUEST_FIELDS,
                )
                writer.writeheader()
                for request in requests:
                    writer.writerow(self._serialize(request))
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, self._path)
        except OSError as error:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise RepositoryError("credit requests CSV could not be updated") from error

    @staticmethod
    def _normalize_request(
        request: CreditRequest,
        status: CreditRequestStatus,
    ) -> CreditRequest:
        return CreditRequest(
            client_cpf=request.client_cpf,
            requested_at=request.requested_at,
            current_limit=request.current_limit.quantize(TWO_DECIMAL_PLACES),
            requested_limit=request.requested_limit.quantize(TWO_DECIMAL_PLACES),
            status=status,
        )

    @staticmethod
    def _identity(
        request: CreditRequest,
    ) -> tuple[str, datetime, Decimal, Decimal]:
        return (
            request.client_cpf,
            request.requested_at,
            request.current_limit,
            request.requested_limit,
        )

    @staticmethod
    def _serialize(request: CreditRequest) -> dict[str, str]:
        return {
            "cpf_cliente": request.client_cpf,
            "data_hora_solicitacao": request.requested_at.isoformat(),
            "limite_atual": f"{request.current_limit:.2f}",
            "novo_limite_solicitado": f"{request.requested_limit:.2f}",
            "status_pedido": request.status.value,
        }
