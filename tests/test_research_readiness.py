from uav_isac.governance import audit_research_readiness
from uav_isac.interfaces.cli import main


def test_repository_readiness_exposes_real_remaining_legacy_boundary():
    status = audit_research_readiness()

    assert not status.ready_for_algorithm_optimization
    assert status.metrics["legacy_adapter_backends"] == 3
    assert status.metrics["ambiguous_result_files"] == 0
    assert status.metrics["portable_formal_assets"] == 2
    assert any("legacy backends" in blocker for blocker in status.blockers)


def test_refactor_status_cli_fails_closed_until_ready(capsys):
    assert main(["refactor-status"]) == 2
    output = capsys.readouterr().out
    assert '"ready_for_algorithm_optimization": false' in output
    assert '"legacy_adapter_backends": 3' in output
