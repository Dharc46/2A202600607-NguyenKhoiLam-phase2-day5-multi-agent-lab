"""Domain-specific errors for the lab skeleton."""


class LabError(Exception):
    """Base error for the lab package."""


class AgentExecutionError(LabError):
    """Raised when an agent fails after retries/fallbacks."""


class ValidationError(LabError):
    """Raised when state or output validation fails."""
