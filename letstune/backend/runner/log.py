import time
from typing import Any, Protocol

__all__ = ["Logger", "NULL_LOGGER", "ConsoleLogger"]


class Logger(Protocol):
    def log(self, **kwargs: Any) -> None:
        ...


class _NullLogger:
    def log(self, **kwargs: Any) -> None:
        pass


NULL_LOGGER: Logger = _NullLogger()


def _format_event(
    parts: list[str],
    kwargs: dict[str, Any],
    category: str,
    id_key: str,
    event: str,
    sub_event: str,
) -> None:
    if id_key in kwargs:
        msg = f"{category} {kwargs.pop(id_key):3}"

        if event == category:
            msg += f" {sub_event}"

        parts.append(msg)


class ConsoleLogger:
    def log(self, **kwargs: Any) -> None:
        event = kwargs.pop("event")
        sub_event = f"{kwargs.pop('sub_event'):>5}"

        parts: list[str] = []

        _format_event(
            parts,
            kwargs,
            "training",
            "training_id",
            event,
            sub_event,
        )
        _format_event(
            parts,
            kwargs,
            "epoch",
            "epoch",
            event,
            sub_event,
        )

        if event not in {"epoch", "training"}:
            parts.append(f"{event} {sub_event}")

        if "status" in kwargs:
            # move "status" to the end
            kwargs["status"] = kwargs.pop("status")

        if kwargs:
            parts.append(", ".join(f"{k}={v}" for k, v in kwargs.items()))

        self.print(": ".join(parts))

    def print(self, s: str) -> None:
        t = time.strftime("%H:%M:%S")
        print(f"[letstune] {t:>8} {s}")
