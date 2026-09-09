import pytest

from scripts import run_mappo
from tools import run_strict_distributed_pilot, run_strict_distributed_sweep
from uav_isac.governance import OperationBlockedError


def test_training_executor_is_blocked_outside_managed_run():
    with pytest.raises(OperationBlockedError):
        run_mappo.main()


def test_full_sweep_entrypoint_requests_result_refresh_gate(monkeypatch):
    monkeypatch.setattr(
        "uav_isac.governance.assert_operation_allowed",
        lambda operation: (_ for _ in ()).throw(RuntimeError(operation)),
    )
    with pytest.raises(RuntimeError, match="full_result_refresh"):
        run_strict_distributed_sweep.main([])


def test_single_pilot_remains_available_for_migration_audit(monkeypatch):
    monkeypatch.setattr(
        "uav_isac.governance.assert_managed_executor",
        lambda entrypoint: (_ for _ in ()).throw(RuntimeError(entrypoint)),
    )
    with pytest.raises(RuntimeError, match="pilot"):
        run_strict_distributed_pilot.main([])
