from pathlib import Path

import pytest

from uav_isac.governance import load_research_programs
from uav_isac.interfaces.cli import main


def test_repository_research_registry_separates_baseline_and_candidates():
    programs = load_research_programs()
    by_name = {program.name: program for program in programs}

    assert by_name["strict_k16q16_baseline"].formal_eligible
    assert by_name["strict_k16q16_baseline"].state == "frozen_baseline"
    assert not by_name["constrained_temporal_power"].formal_eligible
    assert by_name["predictive_constraint_native"].state == "shadow_incomplete"


def test_research_registry_rejects_formal_candidate(tmp_path: Path):
    source = tmp_path / "research.yaml"
    source.write_text(
        """schema_version: 1
programs:
  - name: unsafe
    question: Is an unevaluated idea ready?
    state: candidate
    implementation: tools/idea.py
    profile: config/idea.yaml
    evidence: docs/idea.md
    formal_eligible: true
    next_gate: Run a paired evaluation.
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="frozen_baseline"):
        load_research_programs(source)


def test_research_programs_cli_reports_scientific_state(capsys):
    assert main(["research-programs"]) == 0
    output = capsys.readouterr().out
    assert '"strict_k16q16_baseline"' in output
    assert '"next_gate"' in output
