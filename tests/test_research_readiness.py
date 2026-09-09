from uav_isac.governance import audit_research_readiness
from uav_isac.interfaces.cli import main


def test_repository_readiness_exposes_real_remaining_legacy_boundary():
    status = audit_research_readiness()

    assert status.ready_for_algorithm_optimization
    assert status.metrics["legacy_adapter_backends"] == 0
    assert status.metrics["ambiguous_result_files"] == 0
    assert status.metrics["portable_formal_assets"] == 2
    assert not status.blockers


def test_refactor_status_cli_reports_ready(capsys):
    assert main(["refactor-status"]) == 0
    output = capsys.readouterr().out
    assert '"ready_for_algorithm_optimization": true' in output
    assert '"legacy_adapter_backends": 0' in output
