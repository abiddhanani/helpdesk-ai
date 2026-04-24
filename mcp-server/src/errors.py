class NotFoundError(Exception):
    """Raised when a requested resource does not exist."""


class InvalidInputError(Exception):
    """Raised when input fails validation."""


class BusinessRuleError(Exception):
    """Raised when an operation violates a business rule."""
