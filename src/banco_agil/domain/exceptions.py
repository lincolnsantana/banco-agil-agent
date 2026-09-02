"""Excecoes controladas das fronteiras da aplicacao."""


class BancoAgilError(Exception):
    """Base para falhas esperadas pelo atendimento."""


class DomainError(BancoAgilError):
    """Indica violacao de uma regra de negocio."""


class AuthorizationError(DomainError):
    """Indica tentativa de operacao sem cliente autenticado."""


class RepositoryError(BancoAgilError):
    """Indica falha controlada ao acessar persistencia."""


class IntegrationError(BancoAgilError):
    """Indica falha controlada em uma integracao externa."""


class ExternalServiceUnavailableError(IntegrationError):
    """Indica indisponibilidade temporaria de um servico externo."""
