from typing import Any, Protocol

__all__ = ["Logger", "NULL_LOGGER"]


class Logger(Protocol):
    def log(self, **kwargs: Any) -> None:
        ...


class _NullLogger:
    def log(self, **kwargs: Any) -> None:
        pass


NULL_LOGGER: Logger = _NullLogger()
