"""Application service for dispatching bounded legacy command adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple


@dataclass(frozen=True)
class CommandSpec:
    operation: str
    entrypoint: str
    arguments: Tuple[str, ...] = ()


class ProcessRunner(Protocol):
    def run(self, entrypoint: str, arguments: Tuple[str, ...]) -> int:
        ...


class CommandDispatcher:
    def __init__(self, process_runner: ProcessRunner):
        self._process_runner = process_runner

    def dispatch(self, spec: CommandSpec) -> int:
        if not spec.operation or not spec.entrypoint:
            raise ValueError("command operation and entrypoint must be non-empty")
        return self._process_runner.run(spec.entrypoint, spec.arguments)

