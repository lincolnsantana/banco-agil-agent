"""Persistencia de clientes em CSV com escrita atomica."""

import csv
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from tempfile import NamedTemporaryFile

from filelock import FileLock
from pydantic import ValidationError

from banco_agil.domain.exceptions import RepositoryError
from banco_agil.domain.models import Client

CLIENT_FIELDS = (
    "cpf",
    "data_nascimento",
    "limite_credito",
    "score_credito",
)


def normalize_cpf(cpf: str) -> str:
    """Remove pontuacao usual e valida os 11 digitos do CPF."""
    normalized = re.sub(r"[.\-\s]", "", cpf)
    if not re.fullmatch(r"\d{11}", normalized):
        raise ValueError("CPF must contain exactly 11 digits")
    return normalized


class ClientCsvRepository:
    """Consulta e atualiza clientes em um arquivo CSV validado."""

    def __init__(self, path: Path) -> None:
        """Configura o caminho do CSV sem realizar acesso imediato."""
        self._path = path
        self._lock = FileLock(f"{path}.lock")

    def find_by_cpf(self, cpf: str) -> Client | None:
        """Busca um cliente pelo CPF normalizado.

        Raises:
            RepositoryError: Se o CSV estiver ausente ou invalido.
            ValueError: Se o CPF informado tiver formato invalido.
        """
        normalized_cpf = normalize_cpf(cpf)
        with self._locked():
            clients = self._read_clients()
        return next(
            (client for client in clients if client.cpf == normalized_cpf), None
        )

    def update_credit_score(self, cpf: str, credit_score: int) -> Client:
        """Atualiza o score sob lock e substitui o CSV atomicamente.

        Raises:
            RepositoryError: Se o cliente nao existir ou a persistencia falhar.
            ValueError: Se o CPF informado tiver formato invalido.
            ValidationError: Se o novo score estiver fora do intervalo permitido.
        """
        normalized_cpf = normalize_cpf(cpf)
        with self._locked():
            clients = self._read_clients()
            client_index = self._find_client_index(clients, normalized_cpf)
            current = clients[client_index]
            updated = Client(
                cpf=current.cpf,
                birth_date=current.birth_date,
                credit_limit=current.credit_limit,
                credit_score=credit_score,
            )
            clients[client_index] = updated
            self._write_clients(clients)
        return updated

    def update_credit_limit(self, cpf: str, credit_limit: Decimal) -> Client:
        """Atualiza o limite sob lock e substitui o CSV atomicamente.

        Raises:
            RepositoryError: Se o cliente nao existir ou a persistencia falhar.
            ValueError: Se o CPF informado tiver formato invalido.
            ValidationError: Se o novo limite for invalido.
        """
        normalized_cpf = normalize_cpf(cpf)
        with self._locked():
            clients = self._read_clients()
            client_index = self._find_client_index(clients, normalized_cpf)
            current = clients[client_index]
            updated = Client(
                cpf=current.cpf,
                birth_date=current.birth_date,
                credit_limit=credit_limit,
                credit_score=current.credit_score,
            )
            clients[client_index] = updated
            self._write_clients(clients)
        return updated

    @staticmethod
    def _find_client_index(clients: list[Client], normalized_cpf: str) -> int:
        client_index = next(
            (
                index
                for index, client in enumerate(clients)
                if client.cpf == normalized_cpf
            ),
            None,
        )
        if client_index is None:
            raise RepositoryError("client not found")
        return client_index

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._lock:
            yield

    def _read_clients(self) -> list[Client]:
        try:
            with self._path.open(encoding="utf-8", newline="") as csv_file:
                reader = csv.DictReader(csv_file, strict=True)
                if tuple(reader.fieldnames or ()) != CLIENT_FIELDS:
                    raise RepositoryError("invalid clients CSV: unexpected columns")
                rows = list(reader)
        except OSError as error:
            raise RepositoryError("clients CSV could not be read") from error
        except csv.Error as error:
            raise RepositoryError("invalid clients CSV") from error

        if not rows:
            raise RepositoryError("invalid clients CSV: no client rows")

        clients: list[Client] = []
        seen_cpfs: set[str] = set()
        try:
            for row in rows:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("unexpected row columns")
                client_data: dict[str, object] = {
                    "cpf": normalize_cpf(row["cpf"]),
                    "birth_date": row["data_nascimento"],
                    "credit_limit": row["limite_credito"],
                    "credit_score": row["score_credito"],
                }
                client = Client.model_validate(client_data)
                if client.cpf in seen_cpfs:
                    raise ValueError("duplicate CPF")
                seen_cpfs.add(client.cpf)
                clients.append(client)
        except (KeyError, TypeError, ValueError, ValidationError) as error:
            raise RepositoryError("invalid clients CSV") from error
        return clients

    def _write_clients(self, clients: list[Client]) -> None:
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
                writer = csv.DictWriter(temporary_file, fieldnames=CLIENT_FIELDS)
                writer.writeheader()
                for client in clients:
                    writer.writerow(
                        {
                            "cpf": client.cpf,
                            "data_nascimento": client.birth_date.isoformat(),
                            "limite_credito": format(client.credit_limit, "f"),
                            "score_credito": client.credit_score,
                        }
                    )
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, self._path)
        except OSError as error:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise RepositoryError("clients CSV could not be updated") from error
