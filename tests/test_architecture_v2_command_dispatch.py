import pytest

from uav_isac.application import CommandDispatcher, CommandSpec
from uav_isac.interfaces.cli import main
from uav_isac.governance import OperationBlockedError


class RecordingRunner:
    def __init__(self):
        self.calls = []

    def run(self, entrypoint, arguments):
        self.calls.append((entrypoint, arguments))
        return 7


def test_application_dispatcher_has_no_process_implementation_dependency():
    runner = RecordingRunner()
    status = CommandDispatcher(runner).dispatch(CommandSpec(
        operation="migration_audit",
        entrypoint="pilot",
        arguments=("--seed", "451"),
    ))

    assert status == 7
    assert runner.calls == [("pilot", ("--seed", "451"))]


def test_training_command_requests_algorithm_gate(monkeypatch):
    requested = []

    def stop_at_gate(operation):
        requested.append(operation)
        raise RuntimeError(operation)

    monkeypatch.setattr("uav_isac.interfaces.cli.assert_operation_allowed", stop_at_gate)
    with pytest.raises(RuntimeError, match="algorithm_optimization"):
        main(["train"])
    assert requested == ["algorithm_optimization"]


def test_result_refresh_command_requests_refresh_gate(monkeypatch):
    requested = []

    def stop_at_gate(operation):
        requested.append(operation)
        raise RuntimeError(operation)

    monkeypatch.setattr("uav_isac.interfaces.cli.assert_operation_allowed", stop_at_gate)
    with pytest.raises(RuntimeError, match="full_result_refresh"):
        main(["refresh-results"])
    assert requested == ["full_result_refresh"]
