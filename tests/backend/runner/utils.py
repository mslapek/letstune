from typing import Any


class HistoryLogger:
    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []

    def log(self, **kwargs: Any) -> None:
        self.history.append(kwargs)
